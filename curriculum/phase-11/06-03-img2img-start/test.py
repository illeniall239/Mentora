import math
import unittest

import torch

from solution import img2img_start_step, noise_input_latent


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm())).item()


class TestImg2ImgStart(unittest.TestCase):
    def test_endpoints_of_strength(self):
        self.assertEqual(img2img_start_step(1.0, 50), 50)
        self.assertEqual(img2img_start_step(0.0, 50), 0)
        self.assertEqual(img2img_start_step(1.0, 1), 1)

    def test_floor_not_round(self):
        self.assertEqual(img2img_start_step(0.75, 50), 37)
        self.assertEqual(img2img_start_step(0.25, 50), 12)
        self.assertEqual(img2img_start_step(0.5, 50), 25)
        self.assertEqual(img2img_start_step(0.125, 64), 8)

    def test_start_step_rejects_bad_inputs(self):
        for args in [(1.5, 50), (-0.1, 50), (0.5, 0), (0.5, -3)]:
            with self.assertRaises(ValueError, msg=f"img2img_start_step{args}"):
                img2img_start_step(*args)

    def test_closed_form_matches_by_hand(self):
        latent = torch.tensor([[3.0, 4.0]])
        eps = torch.tensor([[1.0, -1.0]])
        abars = torch.tensor([0.99, 0.64, 0.01])
        self.assertTrue(torch.allclose(noise_input_latent(latent, 1, abars, eps), torch.tensor([[3.0, 2.6]]), atol=1e-5))
        want2 = torch.tensor([[0.1 * 3 + math.sqrt(0.99), 0.1 * 4 - math.sqrt(0.99)]])
        self.assertTrue(torch.allclose(noise_input_latent(latent, 2, abars, eps), want2, atol=1e-5))

    def test_matches_formula_on_a_latent_shaped_tensor(self):
        torch.manual_seed(5)
        latent, eps = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        abars = torch.linspace(0.999, 0.001, 20)
        for step in (0, 7, 19):
            abar = abars[step]
            want = abar.sqrt() * latent + (1 - abar).sqrt() * eps
            got = noise_input_latent(latent, step, abars, eps)
            self.assertEqual(got.shape, latent.shape)
            self.assertTrue(torch.allclose(got, want, atol=1e-5), msg=f"step {step}")
        abar = abars[10]
        got = noise_input_latent(latent, 10, abars, eps)
        self.assertFalse(torch.allclose(got, latent + (1 - abar).sqrt() * eps, atol=1e-2))
        self.assertFalse(torch.allclose(got, abar * latent + (1 - abar) * eps, atol=1e-2))
        self.assertFalse(torch.allclose(got, abar.sqrt() * latent + abar.sqrt() * eps, atol=1e-2))

    def test_low_noise_keeps_the_latent_and_high_noise_approaches_pure_noise(self):
        torch.manual_seed(6)
        latent, eps = torch.randn(2, 16), torch.randn(2, 16)
        abars = torch.tensor([1.0 - 1e-8, 0.5, 1e-8])
        near_latent = noise_input_latent(latent, 0, abars, eps)
        self.assertTrue(torch.allclose(near_latent, latent, atol=1e-3))
        near_noise = noise_input_latent(latent, 2, abars, eps)
        self.assertTrue(torch.allclose(near_noise, eps, atol=1e-3))

    def test_partial_strength_is_not_pure_noise(self):
        torch.manual_seed(7)
        latent, eps = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
        abars = torch.linspace(0.999, 0.001, 50)
        start = img2img_start_step(0.3, 50)
        self.assertEqual(start, 15)
        x = noise_input_latent(latent, start - 1, abars, eps)
        self.assertFalse(torch.allclose(x, eps, atol=1e-2))
        self.assertGreater(cosine(x, latent), 0.7)
        self.assertGreater(cosine(x, latent), cosine(x, eps))

    def test_inputs_untouched_and_bad_arguments_raise(self):
        torch.manual_seed(8)
        latent, eps = torch.randn(3, 3), torch.randn(3, 3)
        l0, e0 = latent.clone(), eps.clone()
        abars = torch.linspace(0.9, 0.1, 4)
        out = noise_input_latent(latent, 2, abars, eps)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(latent, l0))
        self.assertTrue(torch.equal(eps, e0))
        with self.assertRaises(ValueError):
            noise_input_latent(latent, 4, abars, eps)
        with self.assertRaises(ValueError):
            noise_input_latent(latent, -1, abars, eps)
        with self.assertRaises(ValueError):
            noise_input_latent(latent, 1, abars, torch.randn(3, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
