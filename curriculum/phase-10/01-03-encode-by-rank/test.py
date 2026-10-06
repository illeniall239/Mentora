import unittest

from solution import encode

# Ranks 0, 1, 2: (b,c) first, then (a,bc), then (a,b).
DISAGREE = {(98, 99): 256, (97, 256): 257, (97, 98): 258}


class TestEncodeByRank(unittest.TestCase):
    def test_canonical_merges_from_training(self):
        merges = {(97, 97): 256, (256, 97): 257, (257, 98): 258}
        self.assertEqual(encode("aaabdaaabac", merges), [258, 100, 258, 97, 99])

    def test_rank_order_beats_left_to_right(self):
        # Left-to-right would merge (a,b) first and produce [258, 99].
        self.assertEqual(encode("abc", DISAGREE), [257])
        self.assertEqual(encode("abcab", DISAGREE), [257, 258])

    def test_rank_is_dict_order_not_id_order(self):
        # Same pairs, ranks reversed: now (a,b) fires first.
        reversed_ranks = {(97, 98): 258, (97, 256): 257, (98, 99): 256}
        self.assertEqual(encode("abc", reversed_ranks), [258, 99])

    def test_lowest_rank_wins_even_when_it_appears_later_in_the_text(self):
        merges = {(99, 100): 256, (97, 98): 257}
        # (a,b) appears first in the text but (c,d) has the lower rank; the result is the same either
        # way here, so the order is checked through a chain that depends on it below.
        self.assertEqual(encode("abcd", merges), [257, 256])
        chain = {(99, 100): 256, (98, 256): 257, (97, 98): 258}
        self.assertEqual(encode("abcd", chain), [97, 257])

    def test_all_occurrences_merge_before_the_next_rank(self):
        merges = {(97, 98): 256, (256, 99): 257}
        self.assertEqual(encode("abcabc", merges), [257, 257])

    def test_no_overlap_left_to_right(self):
        merges = {(97, 97): 256, (256, 256): 257}
        self.assertEqual(encode("aaa", merges), [256, 97])
        self.assertEqual(encode("aaaa", merges), [257])
        self.assertEqual(encode("aaaaa", merges), [257, 97])

    def test_no_applicable_merges_gives_raw_bytes(self):
        self.assertEqual(encode("xyz", DISAGREE), [120, 121, 122])
        self.assertEqual(encode("abc", {}), [97, 98, 99])
        self.assertEqual(encode("", DISAGREE), [])
        self.assertEqual(encode("🌍", {}), [0xF0, 0x9F, 0x8C, 0x8D])

    def test_multibyte_merges_and_merges_left_untouched(self):
        merges = {(0xF0, 0x9F): 256, (256, 0x8C): 257}
        before = dict(merges)
        self.assertEqual(encode("🌍🌍", merges), [257, 0x8D, 257, 0x8D])
        self.assertEqual(merges, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
