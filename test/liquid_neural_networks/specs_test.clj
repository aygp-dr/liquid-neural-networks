(ns liquid-neural-networks.specs-test
  "Generative checks for every pure s/fdef'd fn, plus data-spec sanity.
  Per https://clojure.org/guides/spec (Testing)."
  (:require [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as gen]
            [clojure.spec.test.alpha :as stest]
            [clojure.test :refer [deftest is testing]]
            [liquid-neural-networks.core :as lnn]
            [liquid-neural-networks.specs :as specs]))

(def ^:private check-opts {:clojure.spec.test.check/opts {:num-tests 50}})

;; Side-effecting fns: fdef'd for instrumentation, never generatively checked.
(def ^:private side-effecting
  #{`lnn/save-network   ; writes a file
    `lnn/load-network   ; reads a file
    `lnn/-main})        ; logs a demo run

(defn- checkable []
  (remove side-effecting (stest/enumerate-namespace 'liquid-neural-networks.core)))

(deftest fdefs-hold-under-generative-testing
  (let [results (stest/check (checkable) check-opts)]
    (is (seq results) "expected at least one fdef'd fn to check")
    (doseq [r results]
      (testing (str (:sym r))
        (is (nil? (:failure r))
            (pr-str (stest/abbrev-result r)))))))

(deftest data-specs-generate-and-conform
  (doseq [k [::specs/dt ::specs/input-vector ::specs/neuron ::specs/layer-config
             ::specs/layer-configs ::specs/network ::specs/sample ::specs/samples]]
    (testing (str k)
      (is (every? (fn [[v _]] (s/valid? k v)) (s/exercise k 10))))))

(deftest generated-values-satisfy-the-core-specs
  (testing "time steps are ::core/dt"
    (is (every? #(s/valid? ::lnn/dt %) (gen/sample (s/gen ::specs/dt) 20))))
  (testing "neuron parameters are ::core/tau, ::core/weight, ::core/bias"
    (doseq [n (gen/sample (s/gen ::specs/neuron) 10)]
      (is (s/valid? ::lnn/tau (:tau n)))
      (is (s/valid? ::lnn/weight (:weights n)))
      (is (s/valid? ::lnn/bias (:bias n))))))

(deftest real-values-conform
  (testing "the -main example network"
    (let [config [{:size 4 :type :ltc :neuron-params {:tau 2.0 :A 1.0}}
                  {:size 2 :type :cfc :neuron-params {:tau 1.5 :A 0.8}}
                  {:size 1 :type :ltc :neuron-params {:tau 1.0 :A 1.0}}]
          network (lnn/create-liquid-network config)
          data [{:input [0.1 0.2 0.3 0.4] :target [0.5]}
                {:input [0.2 0.3 0.4 0.5] :target [0.6]}]]
      (is (s/valid? ::specs/layer-configs config))
      (is (s/valid? ::specs/network network))
      (is (s/valid? ::specs/layer-outputs (lnn/forward-pass network [0.5 0.3 0.8 0.2] 0.1)))
      (is (s/valid? ::specs/samples data))
      (is (s/valid? ::specs/benchmark-result (lnn/benchmark-network network data 0.1)))
      (is (s/valid? ::specs/network-summary (lnn/network-summary network)))))
  (testing "the applications namespace loads"
    (is (some? (requiring-resolve 'liquid-neural-networks.applications/create-application)))))
