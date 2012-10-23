(ns clj-nn.matrices)

;=== AUXILIARY FUNCTIONS ===============================================

(defn unpack
  "Applies identity to the collection, thus unpacking it."
  [coll]
  (apply identity coll))

;=== ACCESSING ELEMENTS ================================================

(defn elem
  "Returns element on the specified position in the matrix.
  Returns nil if the element is not present."
  [matrix row col]
  (get-in matrix [row col]))

;-----------------------------

(defn row
  "Returns nth row of matrix m."
  [matrix n]
  [(get matrix n)])

(defn col
  "Returns nth column of matrix."
  [matrix n]
  [(mapv #(get % n) matrix)])

(defn rowc
  "Returns number of rows in matrix."
  [matrix]
  (count matrix))

(defn colc
  "Returns number of columns in matrix."
  [matrix]
  (count (unpack (row matrix 0))))

;-----------------------------

(defn rows
  "Returns rows of matrix from start to end (inclusive)."
  [matrix start end]
  (subvec matrix start (inc end)))

(defn rows-from
  "Returns rows of matrix starting with nth."
  [matrix n]
  (rows matrix n (dec (rowc matrix))))

(defn rows-to
  "Returns rows of matrix up to nth (inclusive)."
  [matrix n]
  (rows matrix 0 n))

;-----------------------------

(defn cols
  "Returns columns of matrix from start to end (inclusive)."
  [matrix start end]
  (mapv #(unpack (col matrix %)) (range start (inc end))))

(defn cols-from
  "Returns columns of matrix starting with nth."
  [matrix n]
  (cols matrix n (dec (colc matrix))))

(defn cols-to
  "Returns columns of matrix up to nth (inclusive)."
  [matrix n]
  (cols matrix 0 n))

;=== MAP FUNCTIONS =====================================================

(defn emap
  "Returns matrix consisting of the results of applying the function f
  to the set of first elements of each matrix, followed by applying f
  to the set of second elements in each matrix, etc. for all the elements."
 ([f m] (mapv #(mapv f %) m))
 ([f m1 m2] (mapv #(mapv f %1 %2) m1 m2))
 ([f m1 m2 m3] (mapv #(mapv f %1 %2 %3) m1 m2 m3))
 ([f m1 m2 m3 & ms] (apply mapv (fn [& xs] (apply mapv f xs)) m1 m2 m3 ms)))

;=== ELEMENT-WISE OPERATIONS WITH NUMBERS ==============================

(defn nadd
  "Returns matrix with n added to all elements."
  [n matrix]
  (emap #(+ % n) matrix))

(defn nsub
  "Returns matrix with n subtracted from all elements."
  [n matrix]
  (emap #(- % n) matrix))

(defn nmul
  "Returns matrix with all elements multiplied by n."
  [n matrix]
  (emap #(* % n) matrix))

(defn ndiv
  "Returns matrix with all elements divided by n."
  [n matrix]
  (emap #(/ % n) matrix))

(defn npow
  "Returns matrix with all elements divided by n."
  [n matrix]
  (emap #(Math/pow % n) matrix))

;=== ELEMENT-WISE OPERATIONS WITH MATRICES =============================

(defn eadd
  "Returns sum of matrices, element by element."
  ([m] (emap + m))
  ([m1 m2] (emap + m1 m2))
  ([m1 m2 m3] (emap + m1 m2 m3))
  ([m1 m2 m3 & ms] (apply emap + m1 m2 m3 ms)))

(defn esub
  "Returns result of subtraction of the first matrix from the second,
  element by element."
  ([m] (emap - m))
  ([m1 m2] (emap - m1 m2))
  ([m1 m2 m3] (emap - m1 m2 m3))
  ([m1 m2 m3 & ms] (apply emap - m1 m2 m3 ms)))

(defn emul
  "Returns result of element-wise multiplication of the matrices."
  ([m] (emap * m))
  ([m1 m2] (emap * m1 m2))
  ([m1 m2 m3] (emap * m1 m2 m3))
  ([m1 m2 m3 & ms] (apply emap * m1 m2 m3 ms)))

(defn ediv
  "Returns result of element-wise division of the matrices."
  ([m] (emap / m))
  ([m1 m2] (emap / m1 m2))
  ([m1 m2 m3] (emap / m1 m2 m3))
  ([m1 m2 m3 & ms] (apply emap / m1 m2 m3 ms)))

;=== SPECIAL MATRICES ==================================================

(defn zeros
  "Returns matrix of ROWSxCOLS size whose elements are all 0."
  [rows cols]
  (vec (repeat rows (vec (repeat cols 0)))))

(defn ones
  "Returns matrix of ROWSxCOLS size whose elements are all 1."
  [rows cols]
  (vec (repeat rows (vec (repeat cols 1)))))

(defn ident
  "Returns identity matrix of ROWSxCOLS size. Identity matrix has all
  elements equal to 0, except those on the main diagonal, which are
  equal to 1."
  [rows cols]
  (let [z (zeros rows cols)]
    (apply emap + (map #(assoc-in z % 1)
                        (map #(vector % %) (range (rowc z)))))))

(defn randm
  "Returns matrix of size ROWSxCOLS initialized to random values from (0;1)."
  [rows cols]
  (emap #(+ (rand) %) (zeros rows cols)))

;=== MATRIX OPERATIONS =================================================

(defn trans
  "Returns transposed matrix."
  [m]
  {:post [(= (rowc %) (colc m)) (= (colc %) (rowc m))]}
  (cols m 0 (dec (colc m))))

;(defn inv
;  "Returns inverted matrix."
;  [m]
;)

(defn prod
  "Returns product of two matrices."
  [m1 m2]
  {:pre [(= (colc m1) (rowc m2))]
   :post [(= (rowc %) (rowc m1)) (= (colc %) (colc m2))]}
  (let [rows (rowc m1)
        cols (colc m2)
        indexes (for [r (range rows) c (range cols)] [r c])]
    (loop [matrix (zeros rows cols)
           [idx & other] indexes]
      (let [[r c] idx
            val (apply + (unpack (emul (row m1 r) (col m2 c))))
            new (assoc-in matrix idx val)]
        (if-not other
          new
          (recur new other))))))

;=== AUXILIARY FUNCTIONS ===============================================

(defn check
  "Performs various checks on supplied matrix and returns whether all passed."
  [m]
  (and
    (apply = (mapv count m))
    (apply = (mapv count (trans m)))))

;=== FOLD OPERATIONS ===================================================

(defn rowfold
  "Returns matrix consisting of left folds for each row."
  ([f m] (mapv #(vector (reduce f %)) m))
  ([f val m] (mapv #(vector (reduce f val %)) m)))

(defn colfold
  "Returns matrix consisting of upper folds for each column."
  ([f m] (trans (rowfold f (trans m))))
  ([f val m] (trans (rowfold f val (trans m)))))

(defn fold
  "Returns the result of applying (left upper) fold to all elements of matrix."
  ([f m] (reduce f (apply concat m)))
  ([f val m] (reduce f val (apply concat m))))

;=== SUM OPERATIONS ===================================================

(defn rowsum
  "Returns vector of sums for every row of matrix."
  [m]
  (rowfold + m))

(defn colsum
  "Returns vector of sums for every column of matrix."
  [m]
  (colfold + m))

(defn sum
  "Return the sum of all elements in the matrix."
  [m]
  (fold + m))

;=== ADDING ELEMENTS ===================================================

(defn add-under
  "Returns the first matrix with all rows of the second one added to the bottom."
  [m1 m2]
  (vec (concat m1 m2)))

(defn add-besides
  "Returns the first matrix with all columns of the second one added besides."
  [m1 m2]
  (trans (add-under (trans m1) (trans m2))))
