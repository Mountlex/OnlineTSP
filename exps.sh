rm results/*
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/results_s25.csv -s=25
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/results_s75.csv -s=75
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/results_s150.csv -s=150
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/results_s300.csv -s=300


