# Experimental async module DAG state foundation

This isolated foundation is not connected to public or native artifact admission.
It validates a bounded acyclic module dependency graph and module-body start,
await, resume, completion, rejection and dependency-failure propagation. Plans
derive source digests and dependency edges from evaluated-module linker metadata;
start order is derived from observed transitions, not prescribed sibling order.

Runtime frame, execution-Promise, awaited-Promise and reaction-job identities must
agree. Promise/job outcomes must agree for both fulfillment and rejection.
State is checked before and after transitions. Completed dependency edges and
propagated failure origins require settled dependency authority. The caller must
supply authoritative runtime facts; this helper alone cannot establish their
source provenance or own a public event.

The paired probe runs original vm.SourceTextModule behavior, existing owned
scheduler regressions, standalone state controls and an actual Python comparison
at every accepted/refused JavaScript snapshot. The cross-language fixture and
complete logs are retained as workflow artifacts. Before those runs finish,
syntax checks are not execution evidence.

Still required before public admission: independently rederived original-source
lifecycle observations; unique generated source-site/event-group ownership;
public module scheduler operations and active-program plan binding; independent
native replay of those records; actual source/native/token/packing qualification.
Cycles, dynamic import, realms, persistent module restoration, production corpus
exposure and learned quality are outside this first DAG foundation. No optimizer
updates are performed and no acceptance package is closed by these state tests.
