rm results/*
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/results_s100.csv -s=100
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/results_s150.csv -s=150
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/results_s100_wc.csv -s=100 --wc-rd
cargo run --release -- exp1 data/instances_l10 --base-sigma=2 --num-sigma=25 -o results/results_s150_wc.csv -s=150 --wc-rd


cargo run --release -- exp2 data/instances_l10 --pred-frac=0.25 -o results/results_s100_p025.csv -s=100
cargo run --release -- exp2 data/instances_l10 --pred-frac=0.5 -o results/results_s100_p050.csv -s=100
cargo run --release -- exp2 data/instances_l10 --pred-frac=0.75 -o results/results_s100_p075.csv -s=100

