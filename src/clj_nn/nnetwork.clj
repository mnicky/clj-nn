(ns clj-nn.nnetwork
  (:use clj-nn.matrices))

(defn sigmoid
  "Returns value of sigmoid function for supplied 'x'."
  [x]
  (/ 1 (+ 1 (Math/exp (- x)))))

(defn cost
  "Returns the value of the cost function, where
  h - matrix of values returned from feedforward process
  y - matrix of expected values
  m - size of the training set
  theta1, theta2 - parameters of the neural network
  lambda - regularization constant."
  [h y m theta1 theta2 lambda]
  (let [squared #(* % %)
        theta1-1-n-sq (emap squared (trans (cols-from theta1 1)))
        theta2-1-n-sq (emap squared (trans (cols-from theta2 1)))
        regularization (* (/ lambda (* 2 m)) (+ (sum theta1-1-n-sq) (sum theta2-1-n-sq)))
        log #(Math/log %)
        elog #(emap log %)
        one (ones (rowc y) (colc y))]
    (+ (/ (sum (esub (emul (esub y) (elog h)) (emul (esub one y) (elog (esub one h))))) m) regularization)))

(defn add-ones
  "Returns matrix with column of ones added as a first column."
  [m]
  (add-besides (ones (rowc m) 1) m))

(defn feedforward
  "Returns map of computed values of z2, a2, a2-ones z3 and h as keys, where
  x-ones - matrix of training examples, with ones added
  theta1, theta2 - parameters of the neural network.
  For meaninig of outputs, see documentation for the backprop function."
  [x-ones theta1 theta2]
  (let [z2 (prod x-ones (trans theta1))
        a2 (emap sigmoid z2)
        a2-ones (add-ones a2)
        z3 (prod a2-ones (trans theta2))
        h (emap sigmoid z3)]
    {:z2 z2 :a2 a2 :a2-ones a2-ones :z3 z3 :h h}))

(defn backprop
  "Performs one step of backpropagation algorithm and returns
  maps of gradients of theta1 and theta2, where
  h - matrix of values returned from feedforward process
  y - matrix of expected values
  m - size of the training set
  x-ones - matrix of training examples, with ones added
  z2 - helper matrix of outputs of the second (hidden) layer of neural network
       before applying the sigmoid function
  a2-ones - matrix of outputs of the second (hidden) layer of neural network
            after applying the sigmoid function, with ones added
  theta1, theta2 - parameters of the neural network
  lambda - regularization constant."
  [h y m x-ones z2 a2-ones theta1 theta2 lambda]
  (let [d3 (esub h y)
        s (emap sigmoid z2)
        one (ones (rowc s) (colc s))
        sigmoid-grad-z2 (emul s (esub one s))
        theta2-1 (trans (cols-from theta2 1))
        d2 (emul (prod d3 theta2-1) sigmoid-grad-z2)
        D1 (prod (trans d2) x-ones)
        D2 (prod (trans d3) a2-ones)
        D1-0 (trans (cols D1 0 0))
        D2-0 (trans (cols D2 0 0))
        D1-1-n (trans (cols-from D1 1))
        D2-1-n (trans (cols-from D2 1))
        theta1-1-n (trans (cols-from theta1 1))
        theta2-1-n (trans (cols-from theta2 1))
        theta1-grad (add-besides (ndiv m D1-0) (eadd (ndiv m D1-1-n) (nmul (/ lambda m) theta1-1-n)))
        theta2-grad (add-besides (ndiv m D2-0) (eadd (ndiv m D2-1-n) (nmul (/ lambda m) theta2-1-n)))]
    {:theta1-grad theta1-grad
     :theta2-grad theta2-grad}))

(defn grad-descent
  "Performs gradient descent for specified number of steps and returns
  theta values as map with keys :theta1 and :theta2.
  x - matrix of training examples
  y - matrix of expected values
  theta1, theta2 - parameters of the neural network
  alpha - learning rate
  lambda - regularization constant
  steps - number of steps for gradient descent"
  [x y theta1 theta2 alpha lambda steps]
  (let [m (rowc x)
        x-ones (add-ones x)]
    ;(println (cost h y m theta1 theta2 lambda))
    (loop [theta1 theta1
           theta2 theta2
           steps steps]
      (if (zero? steps)
        {:theta1 theta1 :theta2 theta2}
        (let [{:keys [h z2 a2-ones]} (feedforward x-ones theta1 theta2)
              {:keys [theta1-grad theta2-grad]} (backprop h y m x-ones z2 a2-ones theta1 theta2 lambda)]
          (recur (esub theta1 (nmul alpha theta1-grad))
                 (esub theta2 (nmul alpha theta2-grad))
                 (dec steps)))))))
