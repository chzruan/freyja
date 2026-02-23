from freyja.emulators.xi_mm import MatterXiEmulator, HP
import numpy as np

HP = {
    "redshift": 0.25,
    "n_models": 55,
    "r_min": 0.1,
    "r_max": 60.0,
    "n_r_bins": 90,
    "loss_epsilon": 1e-6,
}


def run_training():
    # --- 1. Initialize and Train ---
    print("Initializing Matter Xi GP Emulator...")
    emulator = MatterXiEmulator(checkpoint_path=None, hp=HP)
    save_path = "./matter_xi_gp_z0.25_rmax60_nr90_lr0.020_earlystop.npz"

    # This triggers data loading, preparation, and hyperparameter optimization
    emulator.train(learning_rate=0.020, n_steps=3200, patience=40)
    print("Training completed.")

    # --- 2. Save the Trained Model ---
    emulator.save(save_path)
    print(f"Training complete. Model saved to {save_path}")

    # --- 3. Reloading (Demonstration) ---
    print("\nReloading emulator from disk for testing...")
    emulator = MatterXiEmulator(checkpoint_path=save_path)

    # --- 4. Compare with a model ---
    for imodel in range(53, 65, 1):
        print(f"\nComparing with model {imodel}...")
        emulator.compare(imodel=imodel)


if __name__ == "__main__":
    run_training()
