from freyja.emulators.pk_mm import MatterAlphaEmulator, HP
import numpy as np

HP = {
    "redshift": 0.25,
    "n_models": 59,
    "k_min": 0.03,
    "k_max": 15.0,
    "n_k_bins": 50,
    "loss_epsilon": 1e-6,
}


def run_training():
    # --- 1. Initialize and Train ---
    print("Initializing Matter Alpha GP Emulator...")
    emulator = MatterAlphaEmulator(checkpoint_path=None, hp=HP)

    # This triggers data loading, preparation, and hyperparameter optimization
    emulator.train(learning_rate=0.009, n_steps=3200, patience=20)
    print("Training completed.")

    # --- 2. Save the Trained Model ---
    save_path = "./matter_alpha_gp_z0.25_nk50_lr0.009_earlystop.npz"
    emulator.save(save_path)
    print(f"Training complete. Model saved to {save_path}")

    # --- 3. Reloading (Demonstration) ---
    print("\nReloading emulator from disk for testing...")
    loaded_emulator = MatterAlphaEmulator(checkpoint_path=save_path)

    # --- 4. Make a Prediction ---
    test_cosmo = np.array([0.31, 0.67, 0.82, 0.96])
    k_test = np.logspace(-2, 1, 10)

    alpha, alpha_err = loaded_emulator.predict(test_cosmo, k_test)
    print("Test prediction:")
    for k, a, a_err in zip(k_test, alpha, alpha_err):
        print(f"k={k:.3f}, alpha={a:.3f} +/- {a_err:.3f}")

    # --- 5. Compare with a model ---
    for imodel in range(53, 65, 1):
        print(f"\nComparing with model {imodel}...")
        loaded_emulator.compare(imodel=imodel)


if __name__ == "__main__":
    run_training()
