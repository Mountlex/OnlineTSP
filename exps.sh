cargo run --release -- exp1 data/instances --base-sigma=2 --num-sigma=25 -o results/results_s5.csv -s=5
cargo run --release -- exp1 data/instances --base-sigma=2 --num-sigma=25 -o results/results_s20.csv -s=20
cargo run --release -- exp1 data/instances --base-sigma=2 --num-sigma=25 -o results/results_s50.csv -s=50
cargo run --release -- exp1 data/instances --base-sigma=2 --num-sigma=1 -o results/results_s50_p075.csv -s=50 --pred-frac 0.75
cargo run --release -- exp1 data/instances --base-sigma=2 --num-sigma=1 -o results/results_s50_p050.csv -s=50 --pred-frac 0.5
cargo run --release -- exp1 data/instances --base-sigma=2 --num-sigma=1 -o results/results_s50_p025.csv -s=50 --pred-frac 0.25

