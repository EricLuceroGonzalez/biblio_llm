# Síntesis: temas derivados y vacíos

## Temas detectados

### Tema 1: Genetic Algorithm Applications
- papers: P01, P04, P05, P07, P08, P10, P11, P12
- regla: Papers that explicitly use genetic algorithms (GA) or related metaheuristics for optimization problems.
- evidencia: P01.method=We utilize a mathematical-programming-based planning process as part of a matheuristic which implements a genetic algorithm to improve lecture plans, tutorial plans, and individual timetables.; P04.method=Developed a hybrid Genetic Algorithm (GA) to provide a nurse schedule for all three shifts of a day.; P05.method=A multi-objective genetic algorithm is designed to solve a multiple objective stochastic programming model.; P07.method=We propose the design of two hybrid metaheuristic algorithms based on a genetic algorithm, variable neighborhood search algorithm and the Aquila optimizer.; P08.method=The framework combines resource enumeration, initial crane layout optimization, and an improved genetic algorithm with adaptive operators and tabu-search-based local refinement.; P10.method=Genetic algorithm was used to provide operational solutions for flexible shifts and working hour planning.; P11.method=The study used the Nurse Scheduling Problem model, implemented with Python and Gurobi.; P12.method=The framework uses integer linear programming (ILP).

### Tema 2: Hybrid Optimization Methods
- papers: P02, P03, P06, P09
- regla: Papers that combine genetic algorithms with other optimization techniques or frameworks.
- evidencia: P02.method=Three meta-heuristics (multi-objective Keshtel algorithm, NSGA-II, and multi-objective tabu search) and data envelopment analysis for ranking Pareto solutions are developed.; P03.method=The paper proposes a canonical genetic algorithm as well as two commercial solvers to solve the formulated MINLP models.; P06.method=The model integrated the Dynamic Weighted Distance-Based Outlier Detection (DWD-OD) algorithm with the Multimodal Resource Collaborative Optimization (MRCOF) framework. It used an improved Genetic Simulated Annealing Algorithm (GSAA) in a two-stage optimization strategy.; P09.method=A mathematical model is combined with a genetic algorithm.

### Tema 3: Healthcare and Scheduling Optimization
- papers: P01, P03, P04, P05, P07, P08, P09, P10, P11, P12
- regla: Papers focused on scheduling problems in healthcare, including nurse scheduling, personnel scheduling, and shift planning.
- evidencia: P01.problem=The challenge is to create customized university timetables that balance individual student preferences with collective criteria like workload and idle time, especially with the integration of online courses and flexibility requirements.; P03.problem=The paper addresses the challenge of effective personnel scheduling during the COVID-19 pandemic, particularly in minimizing the risk of infection spread among employees while meeting organizational goals.; P04.problem=The urgent need to modify nurse scheduling to address the fatigue of nurses caring for COVID-19 patients 24 hours a day.; P05.problem=The paper addresses the problem of day surgery scheduling with patient preferences and limited operation rooms, aiming to increase operation efficiency and reduce surgery costs.; P07.problem=The rational allocation of crew members is a strategy to alleviate crew scarcity, but without an appropriate allocation plan, some flights may be canceled due to a lack of pilots in the scheduling period.; P08.problem=The efficiency of rail-mounted gantry crane (RMG) operations directly affects train handling performance in railway logistics terminals.; P09.problem=Nurse scheduling is an NP-hard and highly context-dependent problem that remains unsolved, requiring guidance for practical implementation without costly tools.; P10.problem=The study addresses the need for optimizing nursing shift planning to improve efficiency and fairness in hospital operations, given the 24-hour nature of hospitals and limited nursing staff.; P11.problem=Creating shift schedules for large-scale events, such as university festivals, presents a significant challenge due to the manual process being time-consuming and mentally burdensome, compounded by limited human resources and individuals handling multiple roles.; P12.problem=Nurse scheduling is a complex problem that requires careful consideration of various factors, including skill heterogeneity, unfavourable relationship factors, and expenses.

### Tema 4: Energy Efficiency in Hospital Systems
- papers: P06
- regla: Papers focused on optimizing energy efficiency and resource allocation in hospital systems.
- evidencia: P06.method=The model integrated the Dynamic Weighted Distance-Based Outlier Detection (DWD-OD) algorithm with the Multimodal Resource Collaborative Optimization (MRCOF) framework. It used an improved Genetic Simulated Annealing Algorithm (GSAA) in a two-stage optimization strategy.; P06.problem=The paper addresses engineering challenges in hospital communication and energy-saving systems, including load imbalance, low energy efficiency, and delayed anomaly detection.

### Tema 5: Multi-objective Optimization in Healthcare
- papers: P02
- regla: Papers that explicitly address multi-objective optimization problems in healthcare scheduling.
- evidencia: P02.method=Three meta-heuristics (multi-objective Keshtel algorithm, NSGA-II, and multi-objective tabu search) and data envelopment analysis for ranking Pareto solutions are developed.; P02.problem=Assigning nurses to departments and shifts based on human factors (skill, preference, compatibility) to enhance teamwork and healthcare efficiency.

> Notas: P06 is ambiguous as it integrates multiple methods (genetic algorithm, simulated annealing) and focuses on energy efficiency, which is not a primary theme in other papers. P07 is also somewhat ambiguous as it combines multiple metaheuristics (genetic algorithm, variable neighborhood search, Aquila optimizer).

## Vacíos detectados

- [coverage] Lack of research on dynamic or real-time scheduling using genetic algorithms.
  - evidencia: The records show that genetic algorithms are used for static scheduling problems (P01, P04, P05, P07, P08, P10, P11, P12), but there is no mention of dynamic or real-time scheduling scenarios where schedules need to be adjusted frequently due to unexpected changes.
  - cómo se podría cubrir: Develop genetic algorithm-based models that can handle dynamic scheduling by incorporating real-time data and feedback loops; Investigate the use of online algorithms or reactive scheduling techniques in conjunction with genetic algorithms.
- [method] Insufficient integration of machine learning techniques with genetic algorithms for optimization.
  - evidencia: While genetic algorithms are widely used (P01, P04, P05, P07, P08, P10, P11, P12), there is no record of combining them with machine learning techniques to enhance their performance or adaptability.
  - cómo se podría cubrir: Explore the integration of machine learning, such as neural networks or reinforcement learning, with genetic algorithms to improve solution quality and convergence speed; Investigate the use of predictive analytics to inform genetic algorithm parameters or search strategies in real-time.
- [evidence] Limited empirical validation of hybrid optimization methods in diverse settings.
  - evidencia: Hybrid optimization methods are proposed in several papers (P02, P03, P06, P09), but the validation is often limited to specific contexts or lacks comprehensive empirical testing across different industries or regions.
  - cómo se podría cubrir: Conduct comparative studies of hybrid optimization methods across multiple domains to assess their generalizability and effectiveness; Develop frameworks for benchmarking hybrid optimization techniques to ensure their robustness and reliability.

## Síntesis

The papers address various scheduling and optimization problems, particularly in healthcare, university timetabling, and energy systems.
The main themes include genetic algorithm applications (P01, P04, P05, P07, P08, P10, P11, P12), hybrid optimization methods (P02, P03, P06, P09), and healthcare scheduling (P01, P03, P04, P05, P07, P08, P09, P10, P11, P12).
The evidence shows that genetic algorithms are effective for static scheduling problems (P01, P04, P05, P07, P08, P10, P11, P12) and can be combined with other methods like mathematical programming (P01, P03, P04, P05, P09, P12) or hybrid metaheuristics (P07).
The main gaps include a lack of research on dynamic or real-time scheduling using genetic algorithms, insufficient integration of machine learning techniques with genetic algorithms, and limited empirical validation of hybrid optimization methods in diverse settings.
P01 demonstrates the use of a matheuristic approach combining mathematical programming and genetic algorithms for university timetabling, while P04 highlights the application of hybrid GAs for nurse scheduling during the COVID-19 pandemic.

> Antes de declarar un vacío, comprueba si es real o si lo causa la búsqueda (sinónimos, otras bases de datos, otros venues).
