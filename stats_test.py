import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

# Import your two implementations
# Assuming the files are named as follows:
import training.brax_stats as original_impl
import training.statistics as new_impl


class StatisticsEquivalenceTest(parameterized.TestCase):

    def setUp(self):
        # Use double precision for the test to rule out float accumulation noise
        jax.config.update("jax_enable_x64", True)
        self.key = jax.random.PRNGKey(0)

    def _get_random_batch(self, shape, key):
        return jax.random.normal(key, shape)

    @parameterized.parameters(
        ("welford",),
        ("ema",),
    )
    def test_equivalence_unweighted(self, mode_str):
        """
        Verifies that both implementations produce identical results 
        for standard (unweighted) updates.
        """
        # --- Configuration ---
        batch_size = 32
        feature_dim = 10
        num_updates = 50
        
        # Brax defaults
        std_min = 1e-6
        std_max = 1e6
        # Note: Brax's 'std_eps' is added inside sqrt. 
        # Your new code also adds 'epsilon' inside sqrt.
        # We must ensure they match.
        epsilon = 1e-4 

        # --- 1. Initialize Original (Brax) ---
        dummy_obs = jnp.zeros((feature_dim,))
        
        # Brax init
        orig_state = original_impl.init_state(
            dummy_obs, 
            std_eps=epsilon, 
            mode=mode_str
        )

        # --- 2. Initialize New (NNX) ---
        new_stats = new_impl.RunningStatistics(
            dummy_obs,
            mode=mode_str,
            epsilon=epsilon,
            std_min=std_min,
            std_max=std_max,
            clip=None, # We test stats equality, not output clipping yet
        )

        # --- 3. Run Update Loop ---
        for i in range(num_updates):
            key, subkey = jax.random.split(self.key)
            self.key = key
            
            # Generate Data
            # Shape: (32, 10)
            batch = self._get_random_batch((batch_size, feature_dim), subkey)
            
            # Update Original
            orig_state = original_impl.update(
                orig_state, 
                batch, 
                std_min_value=std_min,
                std_max_value=std_max
            )
            
            # Update New
            new_stats.update(batch)

            # --- 4. Verify Step-by-Step Convergence ---
            # Check Count
            # Brax uses UInt64 custom type, we need to cast to verify
            if isinstance(orig_state.count, jnp.ndarray):
                orig_count = float(orig_state.count)
            else:
                # Handle Brax UInt64
                orig_count = float(orig_state.count.hi) * (2**32) + float(orig_state.count.lo)
            
            np.testing.assert_allclose(
                orig_count, 
                new_stats.count[...],
                err_msg=f"Count mismatch at step {i}"
            )

            # Check Mean
            np.testing.assert_allclose(
                orig_state.mean, 
                new_stats.mean[...], 
                atol=1e-6, 
                err_msg=f"Mean mismatch at step {i}"
            )

            # Check Std
            # Note: We must compare 'std' specifically. 
            # In EMA mode, 'var' storage differs slightly in implementation details
            # but the resulting 'std' should be identical.
            np.testing.assert_allclose(
                orig_state.std, 
                new_stats.std[...], 
                atol=1e-6, 
                err_msg=f"Std mismatch at step {i}"
            )

        # --- 5. Verify Normalization ---
        # Generate a final test batch
        test_batch = jax.random.normal(key, (batch_size, feature_dim))
        
        # Normalize Original
        orig_norm = original_impl.normalize(
            test_batch, 
            orig_state
        )
        
        # Normalize New
        new_norm = new_stats.normalize(test_batch)
        
        np.testing.assert_allclose(
            orig_norm,
            new_norm,
            atol=1e-6,
            err_msg="Final normalization output mismatch"
        )

    def test_structure_handling(self):
        """Verifies that nested structures (dicts) are handled identically."""
        obs = {
            'pos': jnp.zeros((3,)),
            'vel': jnp.zeros((3,)),
        }
        batch = {
            'pos': jnp.ones((10, 3)),
            'vel': jnp.ones((10, 3)) * 5.0,
        }
        
        # Brax
        orig_state = original_impl.init_state(obs)
        orig_state = original_impl.update(orig_state, batch)
        
        # New
        new_stats = new_impl.RunningStatistics(obs)
        new_stats.update(batch)
        
        # Check Pos Mean (should be 1.0)
        np.testing.assert_allclose(
            orig_state.mean['pos'],
            new_stats.mean['pos'][...],
            atol=1e-6
        )
        
        # Check Vel Mean (should be 5.0)
        np.testing.assert_allclose(
            orig_state.mean['vel'],
            new_stats.mean['vel'][...],
            atol=1e-6
        )

if __name__ == '__main__':
    absltest.main()
