(ns liquid-neural-networks.core)

(defn greet
  "Returns a greeting message"
  [name]
  (str "Hello, " name "! Welcome to Liquid Neural Networks."))

(defn -main
  "Main entry point"
  [& args]
  (println (greet (or (first args) "World"))))