---
name: clojure-repl
description: Drive Clojure REPL for exploration. Use when exploring data interactively, debugging, or testing hypotheses. Triggers on Clojure, REPL, Babashka, nREPL, tmux repl.
---

# Clojure REPL Harness

## Quick Start

```bash
# Option 1: Direct nREPL (preferred for agents)
clj -M:dev -m nrepl.cmdline --bind 0.0.0.0 --port 1667 &

# Option 2: Babashka (fast, limited)
bb --nrepl-server 1667 &

# Option 3: tmux fallback
tmux new-session -d -s repl 'clj -M:dev'
```

## Direct nREPL (Recommended)

```clojure
(require '[nrepl.core :as nrepl])
(with-open [conn (nrepl/connect :port 1667)]
  (let [client (nrepl/client conn 5000)
        session (:new-session (first (nrepl/message client {:op "clone"})))]
    (doseq [msg (nrepl/message client {:op "eval" :code "(+ 1 2)" :session session})]
      (when (:value msg) (println "=>" (:value msg))))))
```

## tmux Fallback

```bash
# Send code
tmux send-keys -t repl '(+ 1 2)' Enter

# Wait then capture
sleep 2 && tmux capture-pane -t repl -p | tail -10

# Detect state
tmux capture-pane -t repl -p | tail -3 | grep -qE '=>|user>' && echo clojure || echo shell
```

## Hypothesis Loop

```clojure
;; 1. Observe shape
(keys (first data))
(->> data (map :type) frequencies)

;; 2. Bind intermediate
(def errors (->> data (filter #(= :error (:type %)))))

;; 3. Refine
(->> errors (group-by :source) (map (fn [[k v]] [k (count v)])))
```

## Gotchas

| Problem | Fix |
|---------|-----|
| Output truncated | `sleep 3` before capture |
| Namespace missing | `(require '[ns :as n] :reload)` |
| Wrong prompt | Check state before sending Clojure |
| Quoting issues | Use single quotes in tmux send-keys |
