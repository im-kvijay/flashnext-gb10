#define _GNU_SOURCE  /* qsort_r */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* All pointers remain owned by live CPU tensors in CheckpointShards. */
void flashnext_gather_shards(const uintptr_t *sources, const int64_t *starts,
                            const int64_t *ends, int64_t shards,
                            const int64_t *ids, int64_t count, int64_t valid_rows,
                            size_t width, uint8_t *output, int threads) {
    #pragma omp parallel for num_threads(threads) if(count >= 64 && threads > 1) schedule(static)
    for (int64_t i = 0; i < count; ++i) {
        int64_t id = ids[i];
        uint8_t *destination = output + (size_t)i * width;
        if (id < 0 || id >= valid_rows) {
            memset(destination, 0, width);
            continue;
        }
        int64_t lo = 0, hi = shards;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (starts[mid] <= id) lo = mid + 1;
            else hi = mid;
        }
        int64_t shard = lo - 1;
        if (shard < 0 || id >= ends[shard]) {
            memset(destination, 0, width);
            continue;
        }
        const uint8_t *source = (const uint8_t *)sources[shard];
        memcpy(destination, source + (size_t)(id - starts[shard]) * width, width);
    }
}

/* Same rows read with pread at high queue depth (O_DIRECT when aligned=1). Page-fault gathers
   through a file mapping reach only a few concurrent device reads; one step's
   hundreds of random rows then stall the decoder. fds/bases give each part's
   descriptor and file offset. Returns the number of failed reads. */
#include <stdlib.h>
#include <unistd.h>
int flashnext_pread_shards(const int *fds, const int64_t *bases, const int64_t *starts,
                           const int64_t *ends, int64_t shards, const int64_t *ids,
                           int64_t count, int64_t valid_rows, size_t width,
                           uint8_t *output, int threads, int aligned) {
    int failures = 0;
    #pragma omp parallel num_threads(threads) reduction(+:failures)
    {
        uint8_t *bounce = NULL;
        size_t capacity = ((width + 4095) / 4096 + 1) * 4096;
        if (posix_memalign((void **)&bounce, 4096, capacity)) bounce = NULL;
        #pragma omp for schedule(dynamic, 1)
        for (int64_t i = 0; i < count; ++i) {
            int64_t id = ids[i];
            uint8_t *destination = output + (size_t)i * width;
            int64_t lo = 0, hi = shards;
            while (lo < hi) {
                int64_t mid = lo + (hi - lo) / 2;
                if (starts[mid] <= id) lo = mid + 1;
                else hi = mid;
            }
            int64_t shard = lo - 1;
            if (id < 0 || id >= valid_rows || shard < 0 || id >= ends[shard]) {
                memset(destination, 0, width);
                continue;
            }
            int64_t offset = bases[shard] + (id - starts[shard]) * (int64_t)width;
            if (!aligned) {
                /* Buffered: page-cache hits are copies, misses read in parallel. */
                if (pread(fds[shard], destination, width, offset) != (ssize_t)width) failures++;
                continue;
            }
            int64_t first = offset & ~(int64_t)4095;
            size_t span = (size_t)(((offset + (int64_t)width + 4095) & ~(int64_t)4095) - first);
            /* The last aligned block may extend past end of file: a short
               read is valid when it covers this row. */
            ssize_t got = bounce ? pread(fds[shard], bounce, span, first) : -1;
            if (got < (ssize_t)(offset - first + (int64_t)width)) {
                failures++;
                continue;
            }
            memcpy(destination, bounce + (offset - first), width);
        }
        free(bounce);
    }
    return failures;
}

/* Row cache in front of flashnext_pread_shards. A 4 KiB page cache entry holds
   one useful 160-byte row for random n-gram IDs; this cache stores rows, so
   the same memory holds about 25 times as many. Two-way set associative with
   one recently-used bit per set. A mutex guards lookups and inserts, never
   device reads, so a prefetch and a step's gather can overlap. */
#include <pthread.h>
#include <sys/mman.h>

typedef struct {
    pthread_mutex_t lock;
    int64_t prefetched;  /* rows inserted by flashnext_row_cache_prefetch */
    int64_t sets;
    size_t width;
    int64_t *tags;     /* 2 * sets, -1 = empty */
    uint8_t *recent;   /* sets: way used most recently */
    uint8_t *data;     /* 2 * sets * width */
    int64_t hits, misses, unique_misses;
} flashnext_row_cache;

static void *map_zeroed(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

flashnext_row_cache *flashnext_row_cache_create(int64_t rows, size_t width) {
    flashnext_row_cache *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    pthread_mutex_init(&c->lock, NULL);
    c->sets = rows / 2 > 0 ? rows / 2 : 1;
    c->width = width;
    c->tags = map_zeroed((size_t)c->sets * 2 * sizeof(int64_t));
    c->recent = map_zeroed((size_t)c->sets);
    c->data = map_zeroed((size_t)c->sets * 2 * width);
    if (!c->tags || !c->recent || !c->data) return NULL;
    memset(c->tags, 0xff, (size_t)c->sets * 2 * sizeof(int64_t));
    return c;
}

void flashnext_row_cache_stats(flashnext_row_cache *c, int64_t *out) {
    pthread_mutex_lock(&c->lock);
    out[0] = c->hits; out[1] = c->misses; out[2] = c->unique_misses; out[3] = c->prefetched;
    pthread_mutex_unlock(&c->lock);
}

static inline int64_t cache_set(const flashnext_row_cache *c, int64_t id) {
    uint64_t h = (uint64_t)id * 0x9E3779B97F4A7C15ull;
    return (int64_t)((h >> 17) % (uint64_t)c->sets);
}

static int compare_by_id(const void *a, const void *b, void *ids) {
    int64_t x = ((const int64_t *)ids)[*(const int64_t *)a];
    int64_t y = ((const int64_t *)ids)[*(const int64_t *)b];
    return (x > y) - (x < y);
}

static inline int lookup(flashnext_row_cache *c, int64_t id, uint8_t *destination) {
    int64_t set = cache_set(c, id);
    int way = c->tags[2 * set] == id ? 0 : (c->tags[2 * set + 1] == id ? 1 : -1);
    if (way < 0) return 0;
    if (destination) memcpy(destination, c->data + ((size_t)(2 * set + way)) * c->width, c->width);
    c->recent[set] = (uint8_t)way;
    return 1;
}

static void insert(flashnext_row_cache *c, int64_t id, const uint8_t *row) {
    int64_t set = cache_set(c, id);
    if (c->tags[2 * set] == id || c->tags[2 * set + 1] == id) return;
    int way = c->tags[2 * set] == -1 ? 0 : (c->tags[2 * set + 1] == -1 ? 1 : 1 - c->recent[set]);
    c->tags[2 * set + way] = id;
    memcpy(c->data + ((size_t)(2 * set + way)) * c->width, row, c->width);
    c->recent[set] = (uint8_t)way;
}

/* Sorts the missing positions in order by ID, reads each distinct ID once and
   returns the rows in miss_rows, ordered like miss_id. */
static int read_misses(const int *fds, const int64_t *bases, const int64_t *starts, const int64_t *ends,
                       int64_t shards, const int64_t *ids, int64_t valid_rows, size_t width, int threads,
                       int aligned, int64_t *order, int64_t misses, int64_t *miss_id, int64_t *unique_out,
                       uint8_t *miss_rows) {
    qsort_r(order, misses, sizeof(int64_t), compare_by_id, (void *)ids);
    int64_t unique = 0;
    for (int64_t m = 0; m < misses; ++m) {
        int64_t id = ids[order[m]];
        if (!unique || miss_id[unique - 1] != id) miss_id[unique++] = id;
    }
    *unique_out = unique;
    return flashnext_pread_shards(fds, bases, starts, ends, shards, miss_id, unique, valid_rows, width,
                                  miss_rows, threads, aligned);
}

int flashnext_cached_pread_shards(flashnext_row_cache *c, const int *fds, const int64_t *bases,
                                  const int64_t *starts, const int64_t *ends, int64_t shards,
                                  const int64_t *ids, int64_t count, int64_t valid_rows, size_t width,
                                  uint8_t *output, int threads, int aligned) {
    if (width != c->width) return (int)count;
    int64_t *order = malloc(count * sizeof(int64_t) + 1), *miss_id = malloc(count * sizeof(int64_t) + 1);
    uint8_t *miss_rows = malloc((size_t)count * width + 1);
    if (!order || !miss_id || !miss_rows) { free(order); free(miss_id); free(miss_rows); return (int)count; }
    int64_t misses = 0, hits = 0;
    pthread_mutex_lock(&c->lock);
    for (int64_t i = 0; i < count; ++i) {
        int64_t id = ids[i];
        uint8_t *destination = output + (size_t)i * width;
        if (id < 0 || id >= valid_rows) { memset(destination, 0, width); continue; }
        if (lookup(c, id, destination)) hits++;
        else order[misses++] = i;
    }
    c->hits += hits;
    c->misses += misses;
    pthread_mutex_unlock(&c->lock);
    int failures = 0;
    if (misses) {
        int64_t unique = 0;
        failures = read_misses(fds, bases, starts, ends, shards, ids, valid_rows, width, threads, aligned,
                               order, misses, miss_id, &unique, miss_rows);
        if (!failures) {
            int64_t u = 0;
            for (int64_t m = 0; m < misses; ++m) {
                int64_t i = order[m];
                while (miss_id[u] != ids[i]) ++u;
                memcpy(output + (size_t)i * width, miss_rows + (size_t)u * width, width);
            }
            pthread_mutex_lock(&c->lock);
            c->unique_misses += unique;
            for (u = 0; u < unique; ++u) insert(c, miss_id[u], miss_rows + (size_t)u * width);
            pthread_mutex_unlock(&c->lock);
        }
    }
    free(order); free(miss_id); free(miss_rows);
    return failures;
}

/* Reads rows that are not cached yet into the cache, with no output. Used to
   start a step's reads as soon as its token IDs exist. */
int flashnext_row_cache_prefetch(flashnext_row_cache *c, const int *fds, const int64_t *bases,
                                 const int64_t *starts, const int64_t *ends, int64_t shards,
                                 const int64_t *ids, int64_t count, int64_t valid_rows, size_t width,
                                 int threads, int aligned) {
    if (width != c->width) return (int)count;
    int64_t *order = malloc(count * sizeof(int64_t) + 1), *miss_id = malloc(count * sizeof(int64_t) + 1);
    uint8_t *miss_rows = malloc((size_t)count * width + 1);
    if (!order || !miss_id || !miss_rows) { free(order); free(miss_id); free(miss_rows); return (int)count; }
    int64_t misses = 0;
    pthread_mutex_lock(&c->lock);
    for (int64_t i = 0; i < count; ++i) {
        int64_t id = ids[i];
        if (id >= 0 && id < valid_rows && !lookup(c, id, NULL)) order[misses++] = i;
    }
    pthread_mutex_unlock(&c->lock);
    int failures = 0;
    if (misses) {
        int64_t unique = 0;
        failures = read_misses(fds, bases, starts, ends, shards, ids, valid_rows, width, threads, aligned,
                               order, misses, miss_id, &unique, miss_rows);
        if (!failures) {
            pthread_mutex_lock(&c->lock);
            for (int64_t u = 0; u < unique; ++u) insert(c, miss_id[u], miss_rows + (size_t)u * width);
            c->prefetched += unique;
            pthread_mutex_unlock(&c->lock);
        }
    }
    free(order); free(miss_id); free(miss_rows);
    return failures;
}
