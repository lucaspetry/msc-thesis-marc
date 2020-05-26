python run_cross_val.py data/foursquare_global/foursquare_global_generic.csv \
            --data.folds 5 \
            --results.folder results \
            --embedder.type autoencoder \
            --model.embedding_rate 0.25 \
            --model.merge_type concatenate \
            --model.rnn_type lstm \
            --model.rnn_cells 100 \
            --model.embedding_trainable \
            --model.patience 50 \
            --seed 1234 \
            --save-models \
            --cuda

python run_cross_val.py data/foursquare_global/foursquare_global_generic.csv \
            --data.folds 5 \
            --results.folder results \
            --embedder.type autoencoder \
            --model.embedding_rate 0.50 \
            --model.merge_type concatenate \
            --model.rnn_type lstm \
            --model.rnn_cells 100 \
            --model.embedding_trainable \
            --model.patience 50 \
            --seed 1234 \
            --save-models \
            --cuda

python run_cross_val.py data/foursquare_global/foursquare_global_generic.csv \
            --data.folds 5 \
            --results.folder results \
            --embedder.type autoencoder \
            --model.embedding_rate 0.75 \
            --model.merge_type concatenate \
            --model.rnn_type lstm \
            --model.rnn_cells 100 \
            --model.embedding_trainable \
            --model.patience 50 \
            --seed 1234 \
            --save-models \
            --cuda

python run_cross_val.py data/foursquare_global/foursquare_global_generic.csv \
            --data.folds 5 \
            --results.folder results \
            --embedder.type autoencoder \
            --model.embedding_rate 1.00 \
            --model.merge_type concatenate \
            --model.rnn_type lstm \
            --model.rnn_cells 100 \
            --model.embedding_trainable \
            --model.patience 50 \
            --seed 1234 \
            --save-models \
            --cuda

python run_cross_val.py data/foursquare_global/foursquare_global_generic.csv \
            --data.folds 5 \
            --results.folder results \
            --embedder.type autoencoder \
            --model.embedding_rate 2.00 \
            --model.merge_type concatenate \
            --model.rnn_type lstm \
            --model.rnn_cells 100 \
            --model.embedding_trainable \
            --model.patience 50 \
            --seed 1234 \
            --save-models \
            --cuda

python run_cross_val.py data/foursquare_global/foursquare_global_generic.csv \
            --data.folds 5 \
            --results.folder results \
            --embedder.type autoencoder \
            --model.embedding_rate 5.00 \
            --model.merge_type concatenate \
            --model.rnn_type lstm \
            --model.rnn_cells 100 \
            --model.embedding_trainable \
            --model.patience 50 \
            --seed 1234 \
            --save-models \
            --cuda
