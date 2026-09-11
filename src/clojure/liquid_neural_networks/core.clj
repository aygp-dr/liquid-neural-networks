(ns liquid-neural-networks.core
  "Core implementation of Liquid Neural Networks in Clojure"
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as mop]
            [clojure.core.matrix.linear :as ml]
            [clojure.core.matrix.stats :as ms]
            [clojure.tools.logging :as log]
            [clojure.spec.alpha :as s]
            [clojure.test.check.generators :as gen]
            [fastmath.core :as fm]
            [fastmath.random :as fr]
            [liquid-neural-networks.specs :as specs]))

;; Configure matrix implementation
(m/set-current-implementation :vectorz)

;; =============================================================================
;; Specifications and Validation
;; =============================================================================

(s/def ::positive-number (s/and number? pos?))
(s/def ::non-negative-number (s/and number? (complement neg?)))
(s/def ::weight number?)
(s/def ::bias number?)
(s/def ::tau ::positive-number)
(s/def ::dt ::positive-number)
(s/def ::hidden-state (s/coll-of number?))
(s/def ::input-vector (s/coll-of number?))

;; =============================================================================
;; Mathematical Functions and Utilities
;; =============================================================================

(defn sigmoid
  "Sigmoid activation function with numerical stability"
  [x]
  (let [x (max (min x 500) -500)] ; Clamp to prevent overflow
    (/ 1.0 (+ 1.0 (Math/exp (- x))))))

(s/fdef sigmoid
  :args (s/cat :x ::specs/activation-input)
  :ret (s/and double? #(<= 0.0 % 1.0))
  ;; sigmoid(x) + sigmoid(-x) = 1
  :fn (fn [{{:keys [x]} :args ret :ret}]
        (< (Math/abs (- 1.0 (+ ret (sigmoid (- x))))) 1e-9)))

(defn tanh-stable
  "Numerically stable tanh implementation"
  [x]
  (let [x (max (min x 500) -500)]
    (Math/tanh x)))

(s/fdef tanh-stable
  :args (s/cat :x ::specs/activation-input)
  :ret (s/and double? #(<= -1.0 % 1.0))
  ;; tanh is odd
  :fn (fn [{{:keys [x]} :args ret :ret}]
        (== ret (- (tanh-stable (- x))))))

(defn relu
  "Rectified Linear Unit"
  [x]
  (max 0.0 x))

(s/fdef relu
  :args (s/cat :x ::specs/activation-input)
  :ret (s/and number? (complement neg?)))

(defn leaky-relu
  "Leaky ReLU with configurable slope"
  ([x] (leaky-relu x 0.01))
  ([x alpha]
   (if (pos? x) x (* alpha x))))

(s/fdef leaky-relu
  :args (s/cat :x ::specs/activation-input :alpha (s/? ::specs/unit-open))
  :ret number?
  ;; a positive slope keeps the sign
  :fn (fn [{{:keys [x]} :args ret :ret}]
        (= (pos? x) (pos? ret))))

(defn swish
  "Swish activation function"
  [x]
  (* x (sigmoid x)))

(s/fdef swish
  :args (s/cat :x ::specs/activation-input)
  :ret number?
  ;; swish's global minimum is about -0.2785
  :fn (fn [{ret :ret}]
        (> ret -0.28)))

(defn gelu
  "Gaussian Error Linear Unit (GELU) approximation"
  [x]
  (* 0.5 x (+ 1.0 (Math/tanh (* (Math/sqrt (/ 2.0 Math/PI))
                                (+ x (* 0.044715 (Math/pow x 3))))))))

(s/fdef gelu
  :args (s/cat :x ::specs/activation-input)
  :ret number?
  ;; GELU's global minimum is about -0.17
  :fn (fn [{ret :ret}]
        (> ret -0.18)))

(defn softmax
  "Softmax function for vector"
  [v]
  (let [exp-v (m/emap #(Math/exp (- % (apply max v))) v)
        sum-exp (reduce + exp-v)]
    (m/div exp-v sum-exp)))

(s/fdef softmax
  :args (s/cat :v ::specs/reals)
  :ret (s/coll-of double? :kind vector?)
  ;; a probability distribution over the same number of classes
  :fn (fn [{{:keys [v]} :args ret :ret}]
        (and (= (count v) (count ret))
             (< (Math/abs (- 1.0 (reduce + ret))) 1e-9))))

;; =============================================================================
;; Liquid Time-Constant (LTC) Neuron Implementation
;; =============================================================================

;; Defined below; the neuron records and create-liquid-network call them.
(declare compute-f-function compute-weight-gradient compute-bias-gradient
         compute-tau-gradient create-connectivity-matrix)

(defprotocol LiquidNeuron
  "Protocol for liquid neuron implementations"
  (forward [this hidden-state input dt] "Compute next hidden state")
  (backward [this hidden-state input target dt] "Compute gradients")
  (get-params [this] "Get neuron parameters")
  (set-params [this params] "Set neuron parameters")
  (reset-state [this] "Reset internal state"))

(defrecord LTCNeuron [id weights bias tau A beta activation-fn
                      noise-level learning-rate momentum
                      weight-gradients bias-gradients]
  LiquidNeuron
  (forward [this hidden-state input dt]
    (let [f-val (compute-f-function this hidden-state input)
          noise (when (pos? noise-level)
                  (* noise-level (fr/grand)))
          effective-tau (/ tau (+ 1.0 (* beta (Math/abs f-val))))
          decay-term (/ hidden-state effective-tau)
          drive-term (* f-val A)
          derivative (+ (- drive-term decay-term) (or noise 0.0))
          new-state (+ hidden-state (* dt derivative))]
      (max (min new-state 10.0) -10.0))) ; Bounded stability

  (backward [this hidden-state input target dt]
    (let [prediction (forward this hidden-state input dt)
          error (- target prediction)
          h 1e-6

          ; Compute gradients using finite differences
          weight-grad (compute-weight-gradient this hidden-state input target dt h)
          bias-grad (compute-bias-gradient this hidden-state input target dt h)
          tau-grad (compute-tau-gradient this hidden-state input target dt h)]

      {:error error
       :prediction prediction
       :gradients {:weight weight-grad
                   :bias bias-grad
                   :tau tau-grad}}))

  (get-params [this]
    {:weights weights :bias bias :tau tau :A A :beta beta})

  (set-params [this params]
    (merge this params))

  (reset-state [this]
    (assoc this :weight-gradients 0.0 :bias-gradients 0.0)))

(defn compute-f-function
  "Enhanced f-function with configurable activation"
  [neuron hidden-state input]
  (let [{:keys [weights bias activation-fn]} neuron
        combined-input (+ (* weights (first input)) (* bias hidden-state))]
    (activation-fn combined-input)))

(s/fdef compute-f-function
  :args (s/cat :neuron ::specs/neuron :hidden-state ::specs/real :input ::specs/input-vector)
  :ret number?)

(defn compute-weight-gradient
  "Compute gradient w.r.t. weights using finite differences"
  [neuron hidden-state input target dt h]
  (let [original-weight (:weights neuron)
        neuron-plus (assoc neuron :weights (+ original-weight h))
        neuron-minus (assoc neuron :weights (- original-weight h))
        pred-plus (forward neuron-plus hidden-state input dt)
        pred-minus (forward neuron-minus hidden-state input dt)
        error (- target (forward neuron hidden-state input dt))]
    (* error (/ (- pred-plus pred-minus) (* 2 h)))))

(s/fdef compute-weight-gradient
  :args ::specs/gradient-args
  :ret ::specs/finite)

(defn compute-bias-gradient
  "Compute gradient w.r.t. bias using finite differences"
  [neuron hidden-state input target dt h]
  (let [original-bias (:bias neuron)
        neuron-plus (assoc neuron :bias (+ original-bias h))
        neuron-minus (assoc neuron :bias (- original-bias h))
        pred-plus (forward neuron-plus hidden-state input dt)
        pred-minus (forward neuron-minus hidden-state input dt)
        error (- target (forward neuron hidden-state input dt))]
    (* error (/ (- pred-plus pred-minus) (* 2 h)))))

(s/fdef compute-bias-gradient
  :args ::specs/gradient-args
  :ret ::specs/finite)

(defn compute-tau-gradient
  "Compute gradient w.r.t. tau using finite differences"
  [neuron hidden-state input target dt h]
  (let [original-tau (:tau neuron)
        neuron-plus (assoc neuron :tau (+ original-tau h))
        neuron-minus (assoc neuron :tau (- original-tau h))
        pred-plus (forward neuron-plus hidden-state input dt)
        pred-minus (forward neuron-minus hidden-state input dt)
        error (- target (forward neuron hidden-state input dt))]
    (* error (/ (- pred-plus pred-minus) (* 2 h)))))

(s/fdef compute-tau-gradient
  :args ::specs/gradient-args
  :ret ::specs/finite)

(defn create-ltc-neuron
  "Create a new LTC neuron with specified parameters"
  [id input-size & {:keys [tau A beta activation-fn noise-level learning-rate momentum]
                    :or {tau 1.0 A 1.0 beta 0.1 activation-fn tanh-stable
                         noise-level 0.0 learning-rate 0.01 momentum 0.9}}]
  (->LTCNeuron id
               (fr/grand) ; Random weight
               (* 0.1 (fr/grand)) ; Small random bias
               tau A beta activation-fn noise-level learning-rate momentum
               0.0 0.0)) ; Initialize gradients

(s/fdef create-ltc-neuron
  :args (s/cat :id any? :input-size nat-int? :opts ::specs/neuron-opts)
  :ret ::specs/neuron
  :fn specs/honours-neuron-opts?)

;; =============================================================================
;; Closed-Form Continuous-Time (CfC) Implementation
;; =============================================================================

(defrecord CfCNeuron [id weights bias tau A beta activation-fn
                      noise-level learning-rate]
  LiquidNeuron
  (forward [this hidden-state input dt]
    (let [f-val (compute-f-function this hidden-state input)
          effective-tau (/ tau (+ 1.0 (* beta (Math/abs f-val))))
          decay-factor (Math/exp (- (/ dt effective-tau)))
          target-state (* f-val A)
          noise (when (pos? noise-level)
                  (* noise-level (fr/grand)))
          new-state (+ (* decay-factor hidden-state)
                       (* (- 1.0 decay-factor) target-state)
                       (or noise 0.0))]
      (max (min new-state 10.0) -10.0)))

  (backward [this hidden-state input target dt]
    (let [prediction (forward this hidden-state input dt)
          error (- target prediction)]
      {:error error
       :prediction prediction}))

  (get-params [this]
    {:weights weights :bias bias :tau tau :A A :beta beta})

  (set-params [this params]
    (merge this params))

  (reset-state [this]
    this))

(defn create-cfc-neuron
  "Create a new CfC neuron with specified parameters"
  [id input-size & {:keys [tau A beta activation-fn noise-level learning-rate]
                    :or {tau 1.0 A 1.0 beta 0.1 activation-fn tanh-stable
                         noise-level 0.0 learning-rate 0.01}}]
  (->CfCNeuron id
               (fr/grand)
               (* 0.1 (fr/grand))
               tau A beta activation-fn noise-level learning-rate))

(s/fdef create-cfc-neuron
  :args (s/cat :id any? :input-size nat-int? :opts ::specs/neuron-opts)
  :ret ::specs/neuron
  :fn specs/honours-neuron-opts?)

;; =============================================================================
;; Multi-Layer Liquid Neural Network
;; =============================================================================

(defrecord LiquidNetwork [layers connectivity-matrix global-params])

(defn create-liquid-network
  "Create a multi-layer liquid neural network"
  [layer-configs]
  (let [layers (mapv (fn [config]
                       (let [{:keys [size type neuron-params]} config
                             neuron-fn (case type
                                         :ltc create-ltc-neuron
                                         :cfc create-cfc-neuron)]
                         (mapv #(apply neuron-fn % 1 (flatten (seq neuron-params)))
                               (range size))))
                     layer-configs)
        connectivity (create-connectivity-matrix layers)
        global-params {:learning-rate 0.01
                       :momentum 0.9
                       :weight-decay 0.001
                       :batch-size 32}]
    (->LiquidNetwork layers connectivity global-params)))

(s/fdef create-liquid-network
  :args (s/cat :layer-configs ::specs/layer-configs)
  :ret ::specs/network
  :fn (fn [{{:keys [layer-configs]} :args ret :ret}]
        (= (mapv :size layer-configs) (mapv count (:layers ret)))))

(defn create-connectivity-matrix
  "Create connectivity matrix for network layers"
  [layers]
  (let [total-neurons (reduce + (map count layers))
        matrix (m/zero-matrix total-neurons total-neurons)]
    ; For now, create simple feed-forward connections
    ; TODO: Implement sparse, recurrent, and custom connectivity patterns
    matrix))

(s/fdef create-connectivity-matrix
  :args (s/cat :layers (s/coll-of ::specs/layer :kind vector? :min-count 1 :gen-max 3))
  :ret m/matrix?
  :fn (fn [{{:keys [layers]} :args ret :ret}]
        (let [n (reduce + (map count layers))]
          (= [n n] (vec (m/shape ret))))))

(defn forward-pass
  "Forward pass through the entire network"
  [network input dt]
  (let [layers (:layers network)
        results (atom [])]
    (reduce (fn [current-input layer]
              (let [layer-outputs (mapv (fn [neuron hidden-state]
                                          (forward neuron hidden-state current-input dt))
                                        layer
                                        (take (count layer) (concat (last @results) (repeat 0.0))))]
                (swap! results conj layer-outputs)
                layer-outputs))
            input
            layers)
    @results))

(s/fdef forward-pass
  :args (s/cat :network ::specs/network :input ::specs/input-vector :dt ::specs/dt)
  :ret ::specs/layer-outputs
  ;; one output per neuron, layer by layer
  :fn (fn [{{:keys [network]} :args ret :ret}]
        (= (mapv count (:layers network)) (mapv count ret))))

;; =============================================================================
;; Training and Optimization
;; =============================================================================

(defn compute-loss
  "Compute loss function (MSE for regression, cross-entropy for classification)"
  [predictions targets loss-type]
  (case loss-type
    :mse (/ (reduce + (map #(Math/pow (- %1 %2) 2) predictions targets))
            (count predictions))
    :mae (/ (reduce + (map #(Math/abs (- %1 %2)) predictions targets))
            (count predictions))
    :cross-entropy (- (reduce + (map #(* %1 (Math/log (+ %2 1e-15)))
                                     targets predictions)))))

(s/fdef compute-loss
  :args ::specs/loss-args
  :ret ::specs/finite
  :fn (fn [{{:keys [loss-type]} :args ret :ret}]
        (or (= :cross-entropy loss-type) (>= ret 0))))

(defn sgd-update
  "Stochastic Gradient Descent parameter update"
  [param gradient learning-rate weight-decay]
  (- param (* learning-rate (+ gradient (* weight-decay param)))))

(s/fdef sgd-update
  :args (s/cat :param ::specs/real :gradient ::specs/real
               :learning-rate ::specs/positive :weight-decay ::specs/non-negative)
  :ret number?
  ;; no gradient and no decay leaves the parameter alone
  :fn (fn [{{:keys [param gradient weight-decay]} :args ret :ret}]
        (or (not (and (zero? gradient) (zero? weight-decay)))
            (== param ret))))

(defn adam-update
  "Adam optimizer parameter update"
  [param gradient m v t learning-rate beta1 beta2 epsilon]
  (let [m-new (+ (* beta1 m) (* (- 1 beta1) gradient))
        v-new (+ (* beta2 v) (* (- 1 beta2) (* gradient gradient)))
        m-hat (/ m-new (- 1 (Math/pow beta1 t)))
        v-hat (/ v-new (- 1 (Math/pow beta2 t)))
        param-new (- param (* learning-rate (/ m-hat (+ (Math/sqrt v-hat) epsilon))))]
    {:param param-new :m m-new :v v-new}))

(s/fdef adam-update
  :args (s/cat :param ::specs/real :gradient ::specs/real :m ::specs/real
               :v ::specs/non-negative :t (s/int-in 1 1000)
               :learning-rate ::specs/positive :beta1 ::specs/unit-open
               :beta2 ::specs/unit-open :epsilon ::specs/step)
  :ret ::specs/adam-state)

(defn train-network
  "Train the liquid neural network on a dataset"
  [network training-data epochs dt & {:keys [optimizer loss-type batch-size]
                                      :or {optimizer :adam loss-type :mse batch-size 32}}]
  (let [losses (atom [])]
    (dotimes [epoch epochs]
      (let [epoch-loss (atom 0)
            batches (partition-all batch-size training-data)]
        (doseq [batch batches]
          (let [batch-loss (atom 0)]
            (doseq [{:keys [input target]} batch]
              (let [predictions (forward-pass network input dt)
                    loss (compute-loss (last predictions) target loss-type)]
                (swap! batch-loss + loss)
                ; TODO: Implement backpropagation and parameter updates
                ))
            (swap! epoch-loss + (/ @batch-loss (count batch)))))
        (let [avg-loss (/ @epoch-loss (count batches))]
          (swap! losses conj avg-loss)
          (when (zero? (mod epoch 10))
            (log/info (format "Epoch %d: Loss = %.6f" epoch avg-loss))))))
    {:network network :losses @losses}))

(s/fdef train-network
  :args (s/cat :network ::specs/network :training-data ::specs/samples
               :epochs (s/int-in 0 4) :dt ::specs/dt :opts ::specs/training-opts)
  :ret ::specs/training-result
  ;; one loss per epoch
  :fn (fn [{{:keys [epochs]} :args ret :ret}]
        (= epochs (count (:losses ret)))))

;; =============================================================================
;; Benchmarking and Analysis
;; =============================================================================

(defn benchmark-network
  "Benchmark network performance"
  [network test-data dt]
  (let [start-time (System/nanoTime)
        results (mapv (fn [data]
                        (let [prediction (forward-pass network (:input data) dt)]
                          {:prediction prediction
                           :target (:target data)
                           :error (compute-loss (last prediction) (:target data) :mse)}))
                      test-data)
        end-time (System/nanoTime)
        total-time (/ (- end-time start-time) 1e9)
        avg-error (/ (reduce + (map :error results)) (count results))]
    {:total-time total-time
     :avg-time-per-sample (/ total-time (count test-data))
     :avg-error avg-error
     :throughput (/ (count test-data) total-time)
     :results results}))

(s/fdef benchmark-network
  :args (s/cat :network ::specs/network :test-data ::specs/samples :dt ::specs/dt)
  :ret ::specs/benchmark-result
  :fn (fn [{{:keys [test-data]} :args ret :ret}]
        (= (count test-data) (count (:results ret)))))

(defn analyze-network-dynamics
  "Analyze the dynamical properties of the network"
  [network test-inputs dt]
  (let [stability-metrics (atom [])
        trajectory-data (atom [])]
    (doseq [input test-inputs]
      (let [trajectory (atom [])
            current-state (repeat (count (first (:layers network))) 0.0)]
        ; Simulate trajectory
        (loop [t 0.0
               state current-state]
          (when (< t 10.0) ; Simulate for 10 time units
            (let [next-state (forward-pass network input dt)]
              (swap! trajectory conj {:time t :state state :input input})
              (recur (+ t dt) (last next-state)))))

        ; Analyze stability
        (let [traj @trajectory
              state-norms (map #(Math/sqrt (reduce + (map * (:state %) (:state %)))) traj)
              max-norm (apply max state-norms)
              min-norm (apply min state-norms)
              stability-score (if (< max-norm 50.0) :stable :unstable)]
          (swap! stability-metrics conj {:input input
                                         :max-norm max-norm
                                         :min-norm min-norm
                                         :stability stability-score})
          (swap! trajectory-data conj traj))))

    {:stability-metrics @stability-metrics
     :trajectory-data @trajectory-data
     :overall-stability (if (every? #(= (:stability %) :stable) @stability-metrics)
                          :stable :unstable)}))

(s/fdef analyze-network-dynamics
  :args (s/cat :network ::specs/network
               :test-inputs (s/coll-of ::specs/input-vector :kind vector? :gen-max 3)
               :dt ::specs/dt)
  :ret ::specs/dynamics-result
  :fn (fn [{{:keys [test-inputs]} :args ret :ret}]
        (= (count test-inputs) (count (:stability-metrics ret)))))

;; =============================================================================
;; Utility Functions
;; =============================================================================

(defn save-network
  "Save network to file"
  [network filename]
  (spit filename (pr-str network))
  (log/info (format "Network saved to %s" filename)))

(s/fdef save-network
  :args (s/cat :network ::specs/network :filename string?)
  :ret nil?)

(defn load-network
  "Load network from file"
  [filename]
  (let [network (read-string (slurp filename))]
    (log/info (format "Network loaded from %s" filename))
    network))

(s/fdef load-network
  :args (s/cat :filename string?)
  :ret any?)

(defn network-summary
  "Generate a summary of the network architecture"
  [network]
  (let [layers (:layers network)
        total-neurons (reduce + (map count layers))
        total-params (reduce + (map (fn [layer]
                                      (reduce + (map (fn [neuron]
                                                       (count (get-params neuron)))
                                                     layer)))
                                    layers))]
    {:total-layers (count layers)
     :neurons-per-layer (mapv count layers)
     :total-neurons total-neurons
     :total-parameters total-params
     :connectivity-type "Feed-forward" ; TODO: Make this dynamic
     :global-params (:global-params network)}))

(s/fdef network-summary
  :args (s/cat :network ::specs/network)
  :ret ::specs/network-summary
  :fn (fn [{ret :ret}]
        (= (:total-neurons ret) (reduce + (:neurons-per-layer ret)))))

(defn -main
  "Main entry point for the application"
  [& args]
  (log/info "Starting Liquid Neural Networks application...")

  ; Example usage
  (let [network-config [{:size 4 :type :ltc :neuron-params {:tau 2.0 :A 1.0}}
                        {:size 2 :type :cfc :neuron-params {:tau 1.5 :A 0.8}}
                        {:size 1 :type :ltc :neuron-params {:tau 1.0 :A 1.0}}]
        network (create-liquid-network network-config)
        test-input [0.5 0.3 0.8 0.2]
        dt 0.1]

    (log/info "Network created:" (network-summary network))

    (let [result (forward-pass network test-input dt)]
      (log/info "Forward pass result:" result))

    (let [benchmark-data [{:input [0.1 0.2 0.3 0.4] :target [0.5]}
                          {:input [0.2 0.3 0.4 0.5] :target [0.6]}
                          {:input [0.3 0.4 0.5 0.6] :target [0.7]}]
          benchmark-result (benchmark-network network benchmark-data dt)]
      (log/info "Benchmark results:" benchmark-result))

    (log/info "Application completed successfully.")))

(s/fdef -main
  :args (s/* string?))
