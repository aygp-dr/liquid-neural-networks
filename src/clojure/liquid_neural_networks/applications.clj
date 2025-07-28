(ns liquid-neural-networks.applications
  "Real-world applications of Liquid Neural Networks"
  (:require [liquid-neural-networks.core :as lnn]
            [clojure.core.matrix :as m]
            [clojure.tools.logging :as log]
            [tablecloth.api :as tc]
            [fastmath.random :as fr]))

;; =============================================================================
;; Autonomous Systems Application
;; =============================================================================

(defrecord AutonomousController [lnn sensor-config actuator-config 
                                control-history decision-threshold])

(defn create-autonomous-controller
  "Create an autonomous control system using LNN"
  [sensor-count actuator-count]
  (let [network-config [{:size 8 :type :ltc :neuron-params {:tau 2.0 :A 1.0}}
                        {:size 4 :type :cfc :neuron-params {:tau 1.5 :A 0.8}}
                        {:size actuator-count :type :ltc :neuron-params {:tau 1.0 :A 1.0}}]
        network (lnn/create-liquid-network network-config)]
    (->AutonomousController network
                           {:sensor-count sensor-count
                            :sensor-types [:distance :speed :angle :obstacle]}
                           {:actuator-count actuator-count
                            :actuator-types [:steering :throttle]}
                           []
                           0.5)))

(defn process-sensor-data
  "Process sensor data and generate control commands"
  [controller sensor-data dt]
  (let [normalized-data (mapv #(/ % 255.0) sensor-data)
        control-output (lnn/forward-pass (:lnn controller) normalized-data dt)
        decision (if (> (first (last control-output)) (:decision-threshold controller))
                  :action-required
                  :maintain-state)
        control-command {:timestamp (System/currentTimeMillis)
                        :sensor-data sensor-data
                        :control-output (last control-output)
                        :decision decision}]
    (-> controller
        (update :control-history conj control-command))))

;; =============================================================================
;; Time Series Forecasting Application
;; =============================================================================

(defrecord TimeSeriesPredictor [lnn lookback-window prediction-horizon
                               normalization-params])

(defn create-time-series-predictor
  "Create a time series forecasting system"
  [lookback-window prediction-horizon]
  (let [network-config [{:size 12 :type :ltc :neuron-params {:tau 3.0 :A 1.2}}
                        {:size 8 :type :cfc :neuron-params {:tau 2.0 :A 1.0}}
                        {:size prediction-horizon :type :ltc :neuron-params {:tau 1.0 :A 1.0}}]
        network (lnn/create-liquid-network network-config)]
    (->TimeSeriesPredictor network lookback-window prediction-horizon {})))

(defn normalize-time-series
  "Normalize time series data"
  [data]
  (let [mean-val (/ (reduce + data) (count data))
        std-val (Math/sqrt (/ (reduce + (map #(Math/pow (- % mean-val) 2) data))
                             (count data)))]
    {:normalized (mapv #(/ (- % mean-val) std-val) data)
     :mean mean-val
     :std std-val}))

(defn denormalize-predictions
  "Denormalize predictions back to original scale"
  [predictions mean-val std-val]
  (mapv #(+ (* % std-val) mean-val) predictions))

(defn predict-time-series
  "Predict future values of time series"
  [predictor time-series dt]
  (let [norm-result (normalize-time-series time-series)
        normalized-data (:normalized norm-result)
        sequences (partition (:lookback-window predictor) 1 normalized-data)
        predictions (atom [])]
    (doseq [sequence sequences]
      (let [prediction (lnn/forward-pass (:lnn predictor) (vec sequence) dt)]
        (swap! predictions conj (last prediction))))
    
    {:predictions (denormalize-predictions @predictions 
                                          (:mean norm-result)
                                          (:std norm-result))
     :normalization-params norm-result}))

;; =============================================================================
;; Medical Diagnosis Application
;; =============================================================================

(defrecord MedicalDiagnosisSystem [lnn symptom-encoder diagnosis-decoder
                                  confidence-threshold])

(defn create-medical-diagnosis-system
  "Create a medical diagnosis system using LNN"
  [symptom-count diagnosis-count]
  (let [network-config [{:size 16 :type :ltc :neuron-params {:tau 2.5 :A 1.1}}
                        {:size 12 :type :cfc :neuron-params {:tau 2.0 :A 1.0}}
                        {:size 8 :type :ltc :neuron-params {:tau 1.5 :A 0.9}}
                        {:size diagnosis-count :type :cfc :neuron-params {:tau 1.0 :A 1.0}}]
        network (lnn/create-liquid-network network-config)]
    (->MedicalDiagnosisSystem network {} {} 0.7)))

(defn encode-symptoms
  "Encode patient symptoms into numerical format"
  [symptoms symptom-mapping]
  (mapv #(get symptom-mapping % 0.0) symptoms))

(defn decode-diagnosis
  "Decode network output to diagnosis probabilities"
  [output diagnosis-mapping]
  (let [probs (lnn/softmax output)]
    (mapv (fn [prob diagnosis]
           {:diagnosis diagnosis :probability prob})
          probs diagnosis-mapping)))

(defn diagnose-patient
  "Diagnose patient based on symptoms"
  [system symptoms dt]
  (let [encoded-symptoms (encode-symptoms symptoms (:symptom-encoder system))
        output (lnn/forward-pass (:lnn system) encoded-symptoms dt)
        diagnosis-probs (decode-diagnosis (last output) (:diagnosis-decoder system))
        high-confidence (filter #(> (:probability %) (:confidence-threshold system))
                               diagnosis-probs)]
    {:all-diagnoses diagnosis-probs
     :high-confidence-diagnoses high-confidence
     :timestamp (System/currentTimeMillis)}))

;; =============================================================================
;; Robotics Control Application
;; =============================================================================

(defrecord RobotController [lnn joint-config sensor-config trajectory-planner])

(defn create-robot-controller
  "Create a robot control system"
  [joint-count sensor-count]
  (let [network-config [{:size 20 :type :ltc :neuron-params {:tau 1.5 :A 1.0}}
                        {:size 16 :type :cfc :neuron-params {:tau 1.2 :A 0.9}}
                        {:size 12 :type :ltc :neuron-params {:tau 1.0 :A 0.8}}
                        {:size joint-count :type :cfc :neuron-params {:tau 0.8 :A 1.0}}]
        network (lnn/create-liquid-network network-config)]
    (->RobotController network
                      {:joint-count joint-count
                       :joint-limits [[-180 180] [-90 90] [-180 180]]} ; Example limits
                      {:sensor-count sensor-count
                       :sensor-types [:position :velocity :force :torque]}
                      {})))

(defn compute-joint-commands
  "Compute joint commands for robot"
  [controller sensor-data target-pose dt]
  (let [state-vector (concat sensor-data target-pose)
        joint-outputs (lnn/forward-pass (:lnn controller) state-vector dt)
        joint-commands (mapv (fn [output joint-limits]
                              (let [[min-val max-val] joint-limits
                                    scaled-output (+ min-val (* (+ output 1.0) 0.5 (- max-val min-val)))]
                                (max min-val (min max-val scaled-output))))
                            (last joint-outputs)
                            (get-in controller [:joint-config :joint-limits]))]
    {:joint-commands joint-commands
     :timestamp (System/currentTimeMillis)
     :sensor-data sensor-data
     :target-pose target-pose}))

;; =============================================================================
;; Financial Forecasting Application
;; =============================================================================

(defrecord FinancialPredictor [lnn feature-extractors risk-assessor
                              market-indicators])

(defn create-financial-predictor
  "Create a financial forecasting system"
  [feature-count prediction-horizon]
  (let [network-config [{:size 24 :type :ltc :neuron-params {:tau 4.0 :A 1.2}}
                        {:size 16 :type :cfc :neuron-params {:tau 3.0 :A 1.1}}
                        {:size 12 :type :ltc :neuron-params {:tau 2.0 :A 1.0}}
                        {:size prediction-horizon :type :cfc :neuron-params {:tau 1.0 :A 1.0}}]
        network (lnn/create-liquid-network network-config)]
    (->FinancialPredictor network {} {} {})))

(defn extract-financial-features
  "Extract features from financial data"
  [price-data volume-data indicators]
  (let [returns (mapv (fn [p1 p2] (/ (- p2 p1) p1))
                     price-data (rest price-data))
        volatility (lnn/ms/variance returns)
        moving-avg (/ (reduce + (take-last 10 price-data)) 10)
        volume-avg (/ (reduce + (take-last 10 volume-data)) 10)]
    (concat returns [volatility moving-avg volume-avg] indicators)))

(defn predict-financial-movement
  "Predict financial market movement"
  [predictor market-data dt]
  (let [features (extract-financial-features (:prices market-data)
                                           (:volumes market-data)
                                           (:indicators market-data))
        predictions (lnn/forward-pass (:lnn predictor) features dt)
        movement-probs (lnn/softmax (last predictions))
        direction (if (> (first movement-probs) 0.5) :up :down)
        confidence (Math/abs (- (first movement-probs) 0.5))]
    {:predictions (last predictions)
     :movement-probabilities movement-probs
     :predicted-direction direction
     :confidence confidence
     :timestamp (System/currentTimeMillis)}))

;; =============================================================================
;; Application Factory and Utilities
;; =============================================================================

(defn create-application
  "Factory function to create different types of applications"
  [app-type config]
  (case app-type
    :autonomous-control (create-autonomous-controller (:sensor-count config)
                                                    (:actuator-count config))
    :time-series (create-time-series-predictor (:lookback-window config)
                                             (:prediction-horizon config))
    :medical-diagnosis (create-medical-diagnosis-system (:symptom-count config)
                                                      (:diagnosis-count config))
    :robot-control (create-robot-controller (:joint-count config)
                                          (:sensor-count config))
    :financial-forecasting (create-financial-predictor (:feature-count config)
                                                      (:prediction-horizon config))
    (throw (ex-info "Unknown application type" {:type app-type}))))

(defn benchmark-application
  "Benchmark application performance"
  [app test-data dt]
  (let [start-time (System/nanoTime)
        results (case (type app)
                  AutonomousController
                  (mapv #(process-sensor-data app (:sensor-data %) dt) test-data)
                  
                  TimeSeriesPredictor
                  (mapv #(predict-time-series app (:time-series %) dt) test-data)
                  
                  MedicalDiagnosisSystem
                  (mapv #(diagnose-patient app (:symptoms %) dt) test-data)
                  
                  RobotController
                  (mapv #(compute-joint-commands app (:sensor-data %) (:target-pose %) dt) test-data)
                  
                  FinancialPredictor
                  (mapv #(predict-financial-movement app (:market-data %) dt) test-data))
        end-time (System/nanoTime)
        total-time (/ (- end-time start-time) 1e9)]
    
    {:total-time total-time
     :avg-time-per-sample (/ total-time (count test-data))
     :throughput (/ (count test-data) total-time)
     :results results}))
