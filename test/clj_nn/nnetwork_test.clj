(ns clj-nn.nnetwork-test
  (:use clojure.test
        clj-nn.nnetwork))

(deftest sigmoid-fn-test
  (is (= (sigmoid 0) 0.5))
  (is (= (sigmoid 1000) 1.0))
  (is (= (sigmoid -1000) 0.0)))

(deftest cost-fn-test
  (let [h [[0.2 0.2 0.2] [0.3 0.4 0.5] [0.2 0.3 0.4]]
        y [[1 1 1] [2 3 4] [1 2 3]]
        m 3
        theta1 [[1 2 3] [2 3 4] [5 6 7]]
        theta2 [[8 2 3 3] [4 2 4 4] [3 6 2 2]]
        lambda 1.5]
      (is (= (cost h y m theta1 theta2 lambda) 61.1459606853012))))

(deftest add-ones-test
  (is (= (add-ones [[2 3] [4 5]]) [[1 2 3] [1 4 5]])))

(deftest feedforward-test
  (let [x-ones [[1 2 2] [1 5 4] [1 8 7]]
        theta1 [[1 2 3] [2 3 4] [5 6 7]]
        theta2 [[8 2 3 3] [4 2 4 4] [3 6 2 2]]]
    (is (= (:h (feedforward x-ones theta1 theta2)) [[0.999999887461041 0.9999991684438221 0.9999977394486788]
                                          [0.9999998874648379 0.9999991684719722 0.9999977396757007]
                                          [0.9999998874648379 0.9999991684719722 0.999997739675702]]))))

(deftest backprop-test
  (let [h [[0.2 0.2 0.2] [0.3 0.4 0.5] [0.2 0.3 0.4]]
        y [[1 1 1] [2 3 4] [1 2 3]]
        m 3
        x-ones [[1 2 2] [1 5 4] [1 8 7]]
        z2 [[11 16 31] [23 33 63] [38 54 102]]
        a2-ones [[1 0.99 0.99 0.99]
                 [1 0.99 0.99 1.0]
                 [1 1.0 1.0 1.0]]
        theta1 [[1 2 3] [2 3 4] [5 6 7]]
        theta2 [[8 2 3 3] [4 2 4 4] [3 6 2 2]]
        lambda 1.5]
    (is (= (backprop h y m x-ones z2 a2-ones theta1 theta2 lambda)
           {:theta1-grad [[-4.4537393600080545E-5 0.9999109221752833 1.4999109231877887]
                            [-2.700843935649368E-7 1.499999459831108 1.999999459831143]
                            [-8.260059303210882E-14 2.999999999999835 3.499999999999835]],
            :theta2-grad [[-1.0999999999999999 -0.09166666666666679 0.4083333333333332 0.40266666666666673]
                            [-1.7000000000000002 -0.6886666666666665 0.31133333333333346 0.3026666666666664]
                            [-2.3000000000000003 0.7143333333333337 -1.2856666666666663 -1.297333333333333]]}))))

;TODO (deftest grad-descent-test)
