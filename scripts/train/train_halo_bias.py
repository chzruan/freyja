import numpy as np
import jax.numpy as jnp
from freyja.emulators.halo_bias_diffM import HaloBiasEmulator

HP = {
    # Data Constraints
    "logM_cut_max": 13.9,  # Maximum log mass
    "logM_cut_min": 12.4,
    "r_cut_max": 75.0,  # Maximum scale in fitting halo bias in Mpc/h
    "r_cut_min": 35.0,  # Minimum scale
    "n_models": 59,  # Number of simulation models in the training set
    "loss_epsilon": 1e-9,  # Stability floor for variance
}

# hbe = HaloBiasEmulator(hp=HP)
# hbe.prepare_data()


def run_training():
    # --- 1. Initialize and Train ---
    print("Initializing Halo Bias GP Emulator...")
    emulator = HaloBiasEmulator(hp=HP)

    # This triggers data loading, preparation, and hyperparameter optimization
    emulator.train()
    print("Training completed.")
    # --- 2. Save the Trained Model ---
    save_path = "../../freyja/emulators/checkpoints/halo_bias_gp.npz"
    emulator.save(save_path)
    print(f"Training complete. Model saved to {save_path}")

    # --- 3. Reloading (Demonstration) ---
    # You can initialize a new instance directly from the saved file
    print("\nReloading emulator from disk for testing...")
    loaded_emulator = HaloBiasEmulator(saved_path=save_path)

    # --- 4. Make a Prediction ---
    # Define a test cosmology: [Om0, h, sigma8, ns] (Example values)
    # Ensure these match the order expected by your load_cosmology_wrapper
    test_cosmo = np.array([0.31, 0.67, 0.82, 0.96])

    # Define mass coordinates u and v
    # u = (logM1 + logM2) / 2
    # v = (logM1 - logM2) / 2
    logM1 = 13.5
    logM2 = 13.5
    u_val = (logM1 + logM2) / 2.0
    v_val = (logM1 - logM2) / 2.0

    print(f"Predicting bias for Cosmo={test_cosmo}, logM1={logM1}, logM2={logM2}...")

    bias_pred, bias_std = loaded_emulator.predict(test_cosmo, u_val, v_val)
    for imodel in range(61, 65):
        loaded_emulator.compare_model_prediction(imodel=imodel)
        print(f"Predicted Bias B12: {bias_pred:.4f} +/- {bias_std:.4f}")


run_training()
