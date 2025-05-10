1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

```
source ~/.bashrc
```
```
chmod +x run_unlab_sweep.sh
./run_unlab_sweep.sh

```
3. Run the following code for our main experiments.
```
git checkout -- run_unlab_sweep.sh
git pull origin main                 

```

```
nohup ./run_ssl.sh > logs/ssl_$(date +%F_%H-%M-%S).log 2>&1 &
echo "PID = $!"

```
## Our code is based on Garg's work

https://github.com/dtsip/in-context-learning
