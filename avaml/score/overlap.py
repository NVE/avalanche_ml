import math

MAX = 2500

def calc_overlap(vec1, vec2):
    aspect = aspect_overlap(vec1["aspect"], vec2["aspect"])
    height_args = vec1[3:6].tolist() + vec2[3:6].tolist()
    height = height_overlap(*height_args)
    return aspect * height

def aspect_overlap(aspect1, aspect2):
    aspect1 = int(aspect1, base=2) if aspect1 and not math.isnan(int(aspect1)) else 0
    aspect2 = int(aspect2, base=2) if aspect2 and not math.isnan(int(aspect2)) else 0
    return (8 - bin(aspect1 ^ aspect2).count("1")) / 8

def height_overlap(lev1_max, lev1_min, lev1_fill, lev2_max, lev2_min, lev2_fill):
    [(lev1_max, lev1_min, lev1_fill), (lev2_max, lev2_min, lev2_fill)] = sorted(
        [(lev1_max, lev1_min, lev1_fill), (lev2_max, lev2_min, lev2_fill)],
        key=lambda x: x[2]
    )
    if not lev1_fill or math.isnan(lev1_fill):
        return height_overlap(0, 0, 1, lev2_max, lev2_min, lev2_fill)
    if not lev1_fill or math.isnan(lev2_fill):
        return height_overlap(lev1_max, lev1_min, lev1_fill, 0, 0, 1)
    match = {
        (1, 1): (top_top, bottom_bottom),
        (1, 2): (top_bottom, top_bottom),
        (1, 3): (top_sandwich, bottom_middle),
        (1, 4): (top_middle, bottom_sandwich),
        (2, 2): (bottom_bottom, top_top),
        (2, 3): (bottom_sandwich, top_middle),
        (2, 4): (bottom_middle, top_sandwich),
        (3, 3): (sandwich_sandwich, middle_middle),
        (3, 4): (sandwich_middle, sandwich_middle),
        (4, 4): (middle_middle, sandwich_sandwich),
    }
    func, inverse_func = match[(lev1_fill, lev2_fill)]
    verse = func(lev1_max, lev1_min, lev2_max, lev2_min)
    inverse = inverse_func(lev2_max, lev2_min, lev1_max, lev1_min)
    return verse + inverse


def top_top(top1, _top1, top2, _top2):
    return middle_middle(MAX, top1, MAX, top2)

def top_bottom(top, _top, bottom, _bottom):
    return middle_middle(MAX, top, bottom, 0)

def top_sandwich(top, _top, sand_max, sand_min):
    return 1 - max(top, sand_max) / MAX + top_bottom(top, None, sand_min, None)

def top_middle(top, _top, middle_max, middle_min):
    return middle_middle(MAX, top, middle_max, middle_min)

def bottom_bottom(bottom1, _bottom1, bottom2, _bottom2):
    return middle_middle(bottom1, 0, bottom2, 0)

def bottom_sandwich(bottom, _bottom, sand_max, sand_min):
    return top_bottom(sand_max, None, bottom, None) + min(sand_min, bottom) / MAX

def bottom_middle(bottom, _bottom, middle_max, middle_min):
    return middle_middle(bottom, 0, middle_max, middle_min)

def sandwich_sandwich(sand1_max, sand1_min, sand2_max, sand2_min):
    top = top_sandwich(sand1_max, None, sand2_max, sand2_min)
    bottom = bottom_sandwich(sand1_min, None, sand2_max, sand2_min)
    return top + bottom

def sandwich_middle(sand_max, sand_min, middle_max, middle_min):
    top = middle_middle(MAX, sand_max, middle_max, middle_min)
    bottom = middle_middle(sand_min, 0, middle_max, middle_min)
    return top + bottom

def middle_middle(middle1_max, middle1_min, middle2_max, middle2_min):
    srt = sorted([middle1_max, middle1_min, middle2_max, middle2_min])
    if (middle1_max < middle2_min) != (middle1_min < middle2_max):
        return (srt[2] - srt[1]) / MAX
    else:
        return 0
