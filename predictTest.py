from datetime import datetime

from predict import enter_stock,prep_data, prep_model, peek, plot_raw_hist, test_models, predict_dtr  # predict_knn, predict_cart_and_knn


data = enter_stock("AAPL", datetime(2000, 1, 1), datetime(2021, 2, 14))
prices = prep_data(data)
X,Y,X_train,X_validation,Y_train,Y_validation = prep_model(prices)
# Show models & probability - currently using CART
test_models(X_train,Y_train)
predict_dtr(X, Y, X_train, Y_train, X_validation, Y_validation, 20)
# predict_knn(X,Y,X_train,Y_train,X_validation,Y_validation,60)
# predict_cart_and_knn(X,Y,X_train,Y_train,X_validation,Y_validation,60)

# plot_raw_hist(X,Y)
# test_models(X_train, Y_train)
