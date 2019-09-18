from functions import (load_data, process_text, make_tfidf_features, get_predictions, write_predictions)
import warnings
warnings.simplefilter('ignore')

(train, test) = load_data()

(train, test) = process_text(train, test)

(train, test, tfidf_cols) = make_tfidf_features(train, test)

predictions = get_predictions(train, test, tfidf_cols)

write_predictions(predictions)
