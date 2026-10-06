import unittest

from solution import apply_chat_template, loss_mask


class TestChatTemplateAndLossMask(unittest.TestCase):
    def test_two_turn_template_by_hand(self):
        s = apply_chat_template([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}])
        self.assertEqual(s, "<|user|>\nHi<|end|>\n<|assistant|>\nHello!<|end|>\n")

    def test_system_prompt_comes_first_and_order_is_kept(self):
        s = apply_chat_template([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "3+3?"},
        ])
        self.assertEqual(s, "<|system|>\nBe brief.<|end|>\n<|user|>\n2+2?<|end|>\n<|assistant|>\n4<|end|>\n<|user|>\n3+3?<|end|>\n")
        self.assertEqual(s.count("<|end|>"), 4)

    def test_generation_prompt(self):
        msgs = [{"role": "user", "content": "Hi"}]
        self.assertEqual(apply_chat_template(msgs, add_generation_prompt=True), "<|user|>\nHi<|end|>\n<|assistant|>\n")
        self.assertEqual(apply_chat_template(msgs), "<|user|>\nHi<|end|>\n")
        self.assertEqual(apply_chat_template(msgs, add_generation_prompt=False), apply_chat_template(msgs))

    def test_template_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            apply_chat_template([])
        with self.assertRaises(ValueError):
            apply_chat_template([{"role": "tool", "content": "x"}])

    def test_mask_is_one_only_on_assistant_tokens(self):
        tokens = ["<|user|>", "Hi", "<|end|>", "<|assistant|>", "Hello", "!", "<|end|>"]
        roles = ["user"] * 3 + ["assistant"] * 4
        self.assertEqual(loss_mask(tokens, roles), [0, 0, 0, 0, 1, 1, 1])

    def test_prompt_tokens_are_never_scored(self):
        tokens = ["<|system|>", "Be", "brief", "<|end|>", "<|user|>", "What", "?", "<|end|>", "<|assistant|>", "This", "<|end|>"]
        roles = ["system"] * 4 + ["user"] * 4 + ["assistant"] * 3
        mask = loss_mask(tokens, roles)
        self.assertEqual(mask, [0] * 9 + [1, 1])
        self.assertEqual(len(mask), len(tokens))
        self.assertTrue(all(m in (0, 1) for m in mask))

    def test_assistant_tag_is_prompt_but_assistant_end_is_target(self):
        tokens = ["<|assistant|>", "ok", "<|end|>"]
        self.assertEqual(loss_mask(tokens, ["assistant"] * 3), [0, 1, 1])
        # a user turn's <|end|> is never a target even though the same string is scored in the assistant turn
        self.assertEqual(loss_mask(["<|user|>", "ok", "<|end|>"], ["user"] * 3), [0, 0, 0])

    def test_multi_turn_mask_scores_every_assistant_turn(self):
        tokens = ["<|user|>", "a", "<|end|>", "<|assistant|>", "b", "<|end|>", "<|user|>", "c", "<|end|>", "<|assistant|>", "d", "e", "<|end|>"]
        roles = ["user"] * 3 + ["assistant"] * 3 + ["user"] * 3 + ["assistant"] * 4
        self.assertEqual(loss_mask(tokens, roles), [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1])

    def test_mask_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            loss_mask(["a", "b"], ["user"])
        with self.assertRaises(ValueError):
            loss_mask(["a"], ["robot"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
