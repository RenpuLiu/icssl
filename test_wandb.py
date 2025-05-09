# test_wandb.py
import os
import wandb

print("--- Starting W&B Test Script ---")

# 1. Check Environment Variable
print("WANDB_API_KEY from environment:", os.environ.get("WANDB_API_KEY"))

try:
    # 2. Initialize wandb
    wandb.init(project="test-wandb-connectivity")  # Simple project

    # 3. Log a basic value
    wandb.log({"test_metric": 1.0})
    print("Successfully logged a metric to W&B")

    # 4. Finish the run
    wandb.finish()
    print("Successfully finished the W&B run")

    print("--- W&B Test Passed (Basic) ---")

except Exception as e:
    print(f"--- W&B Test Failed: {e} ---")
    print("Please check your W&B setup (login, API key, network).")

print("--- Ending W&B Test Script ---")
