import random


def split_prompt(prompt):
    parts = prompt.split("\n\n")
    return parts[0], parts[1] if len(parts) > 1 else ""


def ordered_months(src_month, months):
    src = str(src_month).zfill(2)
    rest = sorted({str(m).zfill(2) for m in months} - {src})
    return [src] + rest


def group_by_tile(rows):
    groups = {}
    for r in rows:
        key = (r["city"], r["tile_id"])
        groups.setdefault(key, {})[str(r["month"]).zfill(2)] = r
    return groups


def select_year_samples(rows, src_month, required_months, city=None,
                        tile_id=None, n_per_city=1, seed=42):
    src = str(src_month).zfill(2)
    required = [str(m).zfill(2) for m in required_months]
    groups = group_by_tile(rows)

    def has_all(months):
        return src in months and all(m in months for m in required)

    if city is not None or tile_id is not None:
        for (c, t), months in sorted(groups.items(), key=lambda kv: kv[0]):
            if city is not None and c != city:
                continue
            if tile_id is not None and t != tile_id:
                continue
            if has_all(months):
                return [((c, t), months)]
        return []

    by_city = {}
    for (c, t), months in groups.items():
        if has_all(months):
            by_city.setdefault(c, []).append(((c, t), months))
    rng = random.Random(seed)
    out = []
    for c in sorted(by_city):
        pool = list(by_city[c])
        rng.shuffle(pool)
        out.extend(pool[:n_per_city])
    return out


def select_src_month_tiles(rows, src_month="03"):
    src = str(src_month).zfill(2)
    groups = group_by_tile(rows)
    return [((c, t), months) for (c, t), months in sorted(groups.items())
            if src in months]


def shard_list(items, num_shards, shard_index):
    items = list(items)
    if num_shards <= 1:
        return items
    return items[shard_index::num_shards]


def plan_tiles(rows, src_month="03", city=None, tile_id=None,
               limit=None, num_shards=1, shard_index=0):
    samples = select_src_month_tiles(rows, src_month)
    if city is not None:
        samples = [s for s in samples if s[0][0] == city]
    if tile_id is not None:
        samples = [s for s in samples if s[0][1] == tile_id]
    samples = shard_list(samples, num_shards, shard_index)
    if limit:
        samples = samples[:limit]
    return samples
