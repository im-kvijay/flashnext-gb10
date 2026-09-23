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
   one recently-used bit per set. Not thread-safe: calls must be serialized. */
#include <sys/mman.h>

typedef struct {
    int64_t sets;
    size_t width;
    int64_t *tags;     /* 2 * sets, -1 = empty */
    uint8_t *recent;   /* sets: way used most recently */
    uint8_t *data;     /* 2 * sets * width */
    int64_t hits, misses, unique_misses;
    /* scratch for one call */
    int64_t capacity;
    int64_t *miss_index, *miss_id, *order;
    uint8_t *miss_rows;
} flashnext_row_cache;

static void *map_zeroed(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

flashnext_row_cache *flashnext_row_cache_create(int64_t rows, size_t width) {
    flashnext_row_cache *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->sets = rows / 2 > 0 ? rows / 2 : 1;
    c->width = width;
    c->tags = map_zeroed((size_t)c->sets * 2 * sizeof(int64_t));
    c->recent = map_zeroed((size_t)c->sets);
    c->data = map_zeroed((size_t)c->sets * 2 * width);
    if (!c->tags || !c->recent || !c->data) return NULL;
    memset(c->tags, 0xff, (size_t)c->sets * 2 * sizeof(int64_t));
    return c;
}

void flashnext_row_cache_stats(const flashnext_row_cache *c, int64_t *out) {
    out[0] = c->hits; out[1] = c->misses; out[2] = c->unique_misses;
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

static int grow(flashnext_row_cache *c, int64_t count) {
    if (count <= c->capacity) return 0;
    free(c->miss_index); free(c->miss_id); free(c->order); free(c->miss_rows);
    c->miss_index = malloc(count * sizeof(int64_t));
    c->miss_id = malloc(count * sizeof(int64_t));
    c->order = malloc(count * sizeof(int64_t));
    c->miss_rows = malloc((size_t)count * c->width);
    c->capacity = (c->miss_index && c->miss_id && c->order && c->miss_rows) ? count : 0;
    return c->capacity ? 0 : -1;
}

int flashnext_cached_pread_shards(flashnext_row_cache *c, const int *fds, const int64_t *bases,
                                  const int64_t *starts, const int64_t *ends, int64_t shards,
                                  const int64_t *ids, int64_t count, int64_t valid_rows, size_t width,
                                  uint8_t *output, int threads, int aligned) {
    if (width != c->width || grow(c, count)) return (int)count;
    int64_t misses = 0;
    for (int64_t i = 0; i < count; ++i) {
        int64_t id = ids[i];
        uint8_t *destination = output + (size_t)i * width;
        if (id < 0 || id >= valid_rows) { memset(destination, 0, width); continue; }
        int64_t set = cache_set(c, id);
        int way = c->tags[2 * set] == id ? 0 : (c->tags[2 * set + 1] == id ? 1 : -1);
        if (way >= 0) {
            memcpy(destination, c->data + ((size_t)(2 * set + way)) * width, width);
            c->recent[set] = (uint8_t)way;
            c->hits++;
        } else {
            c->miss_index[misses++] = i;
        }
    }
    c->misses += misses;
    if (!misses) return 0;
    /* Deduplicate: rows repeated within a step (across heads or sequences) are read once. */
    for (int64_t m = 0; m < misses; ++m) c->order[m] = c->miss_index[m];
    qsort_r(c->order, misses, sizeof(int64_t), compare_by_id, (void *)ids);
    int64_t unique = 0;
    for (int64_t m = 0; m < misses; ++m) {
        int64_t id = ids[c->order[m]];
        if (!unique || c->miss_id[unique - 1] != id) c->miss_id[unique++] = id;
    }
    c->unique_misses += unique;
    int failures = flashnext_pread_shards(fds, bases, starts, ends, shards, c->miss_id, unique,
                                          valid_rows, width, c->miss_rows, threads, aligned);
    if (failures) return failures;
    int64_t u = 0;
    for (int64_t m = 0; m < misses; ++m) {
        int64_t i = c->order[m];
        while (c->miss_id[u] != ids[i]) ++u;
        memcpy(output + (size_t)i * width, c->miss_rows + (size_t)u * width, width);
    }
    for (u = 0; u < unique; ++u) {
        int64_t id = c->miss_id[u];
        int64_t set = cache_set(c, id);
        int way = c->tags[2 * set] == -1 ? 0 : (c->tags[2 * set + 1] == -1 ? 1 : 1 - c->recent[set]);
        c->tags[2 * set + way] = id;
        memcpy(c->data + ((size_t)(2 * set + way)) * width, c->miss_rows + (size_t)u * width, width);
        c->recent[set] = (uint8_t)way;
    }
    return 0;
}
