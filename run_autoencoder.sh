python run_cross_val.py data/foursquare_nyc/foursquare_nyc_generic.csv \
            --data.folds 5 \
            --results.folder results \
            --embedder.type autoencoder \
            --model.embedding_rate 0.25 \
            --model.merge_type concatenate \
            --model.rnn_type lstm \
            --model.rnn_cells 100 \
            --model.patience 50 \
            --seed 1234 \
            --cuda

python run_cross_val.py data/foursquare_nyc/foursquare_nyc_generic.csv \
            --data.folds 5 \
            --results.folder results \
            --embedder.type autoencoder \
            --model.embedding_rate 0.50 \
            --model.merge_type concatenate \
            --model.rnn_type lstm \
            --model.rnn_cells 100 \
            --model.patience 50 \
            --seed 1234 \
            --cuda

python run_cross_val.py data/foursquare_nyc/foursquare_nyc_generic.csv \
            --data.folds 5 \
            --results.folder results \
            --embedder.type autoencoder \
            --model.embedding_rate 0.75 \
            --model.merge_type concatenate \
            --model.rnn_type lstm \
            --model.rnn_cells 100 \
            --model.patience 50 \
            --seed 1234 \
            --cuda

python run_cross_val.py data/foursquare_nyc/foursquare_nyc_generic.csv \
            --data.folds 5 \
            --results.folder results \
            --embedder.type autoencoder \
            --model.embedding_rate 1.00 \
            --model.merge_type concatenate \
            --model.rnn_type lstm \
            --model.rnn_cells 100 \
            --model.patience 50 \
            --seed 1234 \
            --cuda
