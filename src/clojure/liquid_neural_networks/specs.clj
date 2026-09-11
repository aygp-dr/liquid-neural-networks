(ns liquid-neural-networks.specs
  "Data specs for liquid-neural-networks (https://clojure.org/guides/spec).

  Builds on the scalar specs in liquid-neural-networks.core (::core/tau,
  ::core/dt, ::core/weight, ...): neurons, layer configs, networks, training
  samples and the results of the network fns. Function specs (s/fdef) live
  next to each defn in core."
  (:require [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as gen]
            [liquid-neural-networks.adam :as-alias adam]
            [liquid-neural-networks.benchmark :as-alias benchmark]
            [liquid-neural-networks.core :as-alias core]
            [liquid-neural-networks.dynamics :as-alias dynamics]
            [liquid-neural-networks.layer-config :as-alias layer-config]
            [liquid-neural-networks.network :as-alias network]
            [liquid-neural-networks.neuron :as-alias neuron]
            [liquid-neural-networks.neuron-params :as-alias neuron-params]
            [liquid-neural-networks.sample :as-alias sample]
            [liquid-neural-networks.summary :as-alias summary]
            [liquid-neural-networks.training :as-alias training]))

;; --- Scalars ---
;; Generators are built by fns, not held in vars: building one loads
;; test.check, which is only on the :dev/:test classpath.

(defn finite?
  "A number that is neither NaN nor infinite."
  [x]
  (and (number? x) (Double/isFinite (double x))))

(defn- gen-real [lo hi]
  (gen/double* {:min lo :max hi :infinite? false :NaN? false}))

(s/def ::finite (s/with-gen finite? #(gen-real -100.0 100.0)))
;; network states and weights
(s/def ::real (s/with-gen finite? #(gen-real -10.0 10.0)))
;; activation fn inputs, including the +/-500 clamp region
(s/def ::activation-input (s/with-gen finite? #(gen-real -1000.0 1000.0)))
(s/def ::positive (s/with-gen (s/and finite? pos?) #(gen-real 1.0e-3 10.0)))
(s/def ::non-negative (s/with-gen (s/and finite? (complement neg?)) #(gen-real 0.0 10.0)))
(s/def ::unit-open (s/with-gen (s/and finite? #(< 0 % 1)) #(gen-real 0.01 0.99)))
(s/def ::probability (s/with-gen (s/and finite? #(<= 0 % 1)) #(gen-real 0.0 1.0)))
;; a time step: positive, and at most 1 so a simulation stays short
(s/def ::dt (s/with-gen (s/and finite? pos? #(<= % 1.0)) #(gen-real 0.05 1.0)))
(s/def ::step (s/with-gen (s/and finite? pos?) #(gen-real 1.0e-6 1.0e-3)))

(s/def ::reals (s/coll-of ::real :kind vector? :min-count 1 :gen-max 6))
(s/def ::input-vector
  (s/with-gen (s/and ::core/input-vector seq)
    #(gen/vector (gen-real -1.0 1.0) 1 6)))

;; --- Neurons (the LTCNeuron and CfCNeuron records) ---

;; core registers ::core/* after requiring this ns, so refer to them through
;; s/and, which resolves lazily (a bare alias would resolve at load time).
(s/def ::neuron/weights (s/and ::core/weight))
(s/def ::neuron/bias (s/and ::core/bias))
(s/def ::neuron/tau (s/and ::core/tau))
(s/def ::neuron/A number?)
(s/def ::neuron/beta number?)
(s/def ::neuron/activation-fn ifn?)
(s/def ::neuron/noise-level (s/and ::core/non-negative-number))

(s/def ::neuron-params/tau (s/with-gen (s/and finite? pos?) #(gen-real 0.1 5.0)))
(s/def ::neuron-params/A (s/with-gen finite? #(gen-real 0.1 2.0)))
(s/def ::neuron-params/beta (s/with-gen (s/and finite? (complement neg?)) #(gen-real 0.0 1.0)))
(s/def ::neuron-params/learning-rate ::positive)
(s/def ::neuron-params/momentum ::probability)
(s/def ::neuron-params
  (s/keys :opt-un [::neuron-params/tau ::neuron-params/A ::neuron-params/beta]))

;; The keyword options create-ltc-neuron / create-cfc-neuron accept.
(s/def ::neuron-opts
  (s/with-gen
    (s/keys* :opt-un [::neuron-params/tau ::neuron-params/A ::neuron-params/beta
                      ::neuron/activation-fn ::neuron/noise-level
                      ::neuron-params/learning-rate ::neuron-params/momentum])
    #(gen/fmap (fn [m] (mapcat identity m)) (s/gen ::neuron-params))))

(defn- gen-neuron []
  (gen/fmap (fn [[kind id params]]
              (apply (requiring-resolve (if (= kind :ltc)
                                          'liquid-neural-networks.core/create-ltc-neuron
                                          'liquid-neural-networks.core/create-cfc-neuron))
                     id 1 (mapcat identity params)))
            (gen/tuple (gen/elements [:ltc :cfc]) (gen/choose 0 100) (s/gen ::neuron-params))))

(s/def ::neuron
  (s/with-gen
    (s/keys :req-un [::neuron/weights ::neuron/bias ::neuron/tau ::neuron/A ::neuron/beta
                     ::neuron/activation-fn ::neuron/noise-level])
    gen-neuron))

(s/def ::layer (s/coll-of ::neuron :kind vector? :min-count 1 :gen-max 4))

;; compute-weight-gradient & co.: finite differences with step h
(s/def ::gradient-args
  (s/cat :neuron ::neuron :hidden-state ::real :input ::input-vector
         :target ::real :dt ::dt :h ::step))

(defn honours-neuron-opts?
  "s/fdef :fn for the neuron constructors: a given :tau, :A or :beta is used."
  [{{:keys [opts]} :args ret :ret}]
  (every? (fn [[k v]] (= v (get ret k))) (select-keys opts [:tau :A :beta])))

;; --- Networks ---

(s/def ::layer-config/size (s/int-in 1 6))
(s/def ::layer-config/type #{:ltc :cfc})
(s/def ::layer-config/neuron-params ::neuron-params)
(s/def ::layer-config
  (s/keys :req-un [::layer-config/size ::layer-config/type]
          :opt-un [::layer-config/neuron-params]))
(s/def ::layer-configs (s/coll-of ::layer-config :kind vector? :min-count 1 :max-count 4))

(s/def ::network/layers (s/coll-of ::layer :kind vector? :min-count 1))
(s/def ::network/connectivity-matrix some?)
(s/def ::network/global-params map?)

(defn- gen-network []
  (gen/fmap (fn [cfg] ((requiring-resolve 'liquid-neural-networks.core/create-liquid-network) cfg))
            (s/gen ::layer-configs)))

(s/def ::network
  (s/with-gen
    (s/keys :req-un [::network/layers ::network/connectivity-matrix ::network/global-params])
    gen-network))

;; forward-pass: one output vector per layer
(s/def ::layer-outputs (s/coll-of (s/coll-of ::finite :kind vector?) :kind vector?))

;; --- Data and results ---

(s/def ::sample/input ::input-vector)
(s/def ::sample/target ::reals)
(s/def ::sample (s/keys :req-un [::sample/input ::sample/target]))
(s/def ::samples (s/coll-of ::sample :kind vector? :min-count 1 :gen-max 5))

(s/def ::loss-type #{:mse :mae :cross-entropy})

;; compute-loss: equal-length vectors; cross-entropy wants probabilities.
(s/def ::loss-args
  (s/with-gen
    (s/and (s/cat :predictions ::reals :targets ::reals :loss-type ::loss-type)
           (fn [{:keys [predictions targets loss-type]}]
             (and (= (count predictions) (count targets))
                  (or (not= :cross-entropy loss-type)
                      (every? #(<= 0 % 1) predictions)))))
    #(gen/bind (gen/tuple (gen/choose 1 6) (s/gen ::loss-type))
               (fn [[n t]]
                 (gen/tuple (gen/vector (if (= t :cross-entropy) (gen-real 0.0 1.0) (gen-real -10.0 10.0)) n)
                            (gen/vector (gen-real 0.0 1.0) n)
                            (gen/return t))))))

(s/def ::adam/param number?)
(s/def ::adam/m number?)
(s/def ::adam/v (s/and number? (complement neg?)))
(s/def ::adam-state (s/keys :req-un [::adam/param ::adam/m ::adam/v]))

(s/def ::training/optimizer #{:adam :sgd})
(s/def ::training/loss-type ::loss-type)
(s/def ::training/batch-size (s/int-in 1 40))
(s/def ::training-opts
  (s/with-gen
    (s/keys* :opt-un [::training/optimizer ::training/loss-type ::training/batch-size])
    ;; LTC outputs span [-10, 10], so generated runs keep the default :mse loss
    #(gen/fmap (fn [m] (mapcat identity m)) (s/gen (s/keys :opt-un [::training/batch-size])))))

(s/def ::training/network ::network)
(s/def ::training/losses (s/coll-of number? :kind vector?))
(s/def ::training-result (s/keys :req-un [::training/network ::training/losses]))

(s/def ::benchmark/total-time number?)
(s/def ::benchmark/avg-time-per-sample number?)
(s/def ::benchmark/avg-error number?)
(s/def ::benchmark/throughput number?)
(s/def ::benchmark/results (s/coll-of map? :kind vector?))
(s/def ::benchmark-result
  (s/keys :req-un [::benchmark/total-time ::benchmark/avg-time-per-sample
                   ::benchmark/avg-error ::benchmark/throughput ::benchmark/results]))

(s/def ::dynamics/stability #{:stable :unstable})
(s/def ::dynamics/stability-metrics (s/coll-of map? :kind vector?))
(s/def ::dynamics/trajectory-data (s/coll-of sequential? :kind vector?))
(s/def ::dynamics/overall-stability ::dynamics/stability)
(s/def ::dynamics-result
  (s/keys :req-un [::dynamics/stability-metrics ::dynamics/trajectory-data
                   ::dynamics/overall-stability]))

(s/def ::summary/total-layers pos-int?)
(s/def ::summary/neurons-per-layer (s/coll-of pos-int? :kind vector?))
(s/def ::summary/total-neurons pos-int?)
(s/def ::summary/total-parameters nat-int?)
(s/def ::summary/connectivity-type string?)
(s/def ::summary/global-params map?)
(s/def ::network-summary
  (s/keys :req-un [::summary/total-layers ::summary/neurons-per-layer ::summary/total-neurons
                   ::summary/total-parameters ::summary/connectivity-type
                   ::summary/global-params]))
