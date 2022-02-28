rm results/*
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/results_s50.csv -s=50
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/results_s100.csv -s=100
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/results_s150.csv -s=150


