# Lecture Summaries (L5-L8)

## L5: Randomized Controlled Trial

### Key Concepts

**Causal Effects and the Fundamental Problem**
- A causal effect compares two states: actual vs. counterfactual (what would have happened without treatment)
- The fundamental problem: we cannot observe both states simultaneously for the same unit
- Each unit has two potential outcomes: Y¹ (with treatment) and Y⁰ (without treatment)

**Treatment Effect Estimators**
- **ATE (Average Treatment Effect)**: Average effect for the whole population
- **ATT (Average Treatment on Treated)**: Average effect for those who received treatment
- **ATU (Average Treatment on Untreated)**: Average effect for the control group
- In observational settings, ATT ≠ ATU typically

**Bias Decomposition**
The Simple Difference in Outcomes (SDO) contains:
1. Average Treatment Effect (parameter of interest)
2. Selection Bias: E[Y⁰|D=1] - E[Y⁰|D=0] (differences between groups even without treatment)
3. Heterogeneous Treatment Effect Bias: (1-π)(ATT-ATU) (different treatment effects across groups)

**Independence Assumption**
- Eliminates both selection bias and heterogeneous treatment effect bias
- Requires random assignment of treatment independent of potential outcomes
- When this holds: SDO = ATE (we can "observe the unobservable")

**SUTVA (Stable Unit Treatment Value Assumption)**
- Each treated unit receives the same dose
- No spillovers/externalities between units
- No general equilibrium effects

**Statistical Significance**
- P-value: probability of obtaining results at least as extreme as observed, assuming null hypothesis is true
- P-value is evidence, NOT definitive proof
- P-value does NOT indicate: probability null is true, size of effect, or automatic significance at p<0.05
- Alternatives: Confidence intervals, Bayesian inference

**Sample Size Considerations**
- Fixed vs. adaptive sample size testing: "peeking" creates selection bias
- Pre-determine sample size based on: significance level, marginal error, power, population size

**RCT Guidelines**
1. Define research question and hypotheses
2. Ensure Independence Assumption and SUTVA hold
3. Determine sample size
4. Conduct experiment with random allocation
5. Analyze and report results

**Practical Example**
- Research question: Does framing grading guidelines as bonus vs. penalty affect submission times?
- Sample: 135 students (68 bonus group, 67 penalty group)
- Result: No statistically significant difference (p=0.53)
- Lesson: "No significance" ≠ "no effect" - careful design matters

---

## L7: Price Discrimination and Monopoly - Linear Pricing

### Introduction to Price Discrimination
**What is Price Discrimination?**
- Selling the same product at different prices to different customers
- Examples: Prescription drugs (cheaper in Canada), textbooks (cheaper in UK)
- Presumed to be profitable but not necessarily bad for efficiency
- Question: Is it necessarily bad even if not "fair"?

**Feasibility Requirements**
Two main challenges:
1. **Identification**: Firm must identify demands of different consumer types/markets
2. **Arbitrage prevention**: Stop low-price buyers from reselling to high-price buyers

**Types of Price Discrimination**
- First-degree (personalized pricing)
- Second-degree (menu pricing)
- Third-degree (group pricing) ← Focus of this lecture

### Third-Degree Price Discrimination
**Key Characteristics**
- Consumers differ by observable characteristics
- Uniform price charged within each group
- Different uniform prices across groups
- Examples: Kids eat free, airline fares, early-bird specials, student discounts

**Pricing Rule**
- Low elasticity consumers → High price
- High elasticity consumers → Low price
- Assumes demand functions already estimated via 2SLS, DID, or RCT

### Profit Optimization Framework
**Basic Model**
- Profit: π(Q) = P(Q)Q - F - CQ
- First order condition: MR - MC = 0 → MR = MC
- This is the main pricing condition throughout the course

**What Could Go Wrong?**
- Price function P(Q) may be misspecified
- Marginal cost C may be incorrect
- Model risk assessment is crucial

### Amusement Park Example
**Setup**
- Adults: P_A = 36 - 4Q_A
- Children: P_C = 24 - 4Q_C
- Marginal cost: MC = $4

**No Price Discrimination (NPD)**
- Aggregate demand from both markets
- Find MR for aggregate demand
- Set MR = MC to find optimal quantity
- Result: P = $17, Q = 6.5 million (Q_A = 4.75M, Q_C = 1.75M)
- Profit = $84.5 million

**With Price Discrimination**
- Treat each market separately
- Adults: Q_A = 4, P_A = $20
- Children: Q_C = 2.5, P_C = $14
- Total quantity still 6.5 million (same as NPD with linear demand)
- Profit = $89 million (5.33% increase)

**Key Insight with Linear Demand**
- Price discrimination yields same aggregate output as no discrimination
- But increases profit by redistributing sales

### Non-Constant Marginal Cost
When MC increases (e.g., MC = 0.75 + Q/2):

**No Price Discrimination Procedure**
1. Calculate aggregate demand
2. Calculate associated MR
3. Equate MR with MC for aggregate output
4. Identify price from aggregate demand
5. Identify individual market demands

**With Price Discrimination Procedure**
1. Identify MR in each market
2. Aggregate the MRs
3. Equate aggregate MR with MC for total output
4. Find equilibrium MR
5. Equate this MR with MC in each market for individual quantities
6. Get prices from individual demand curves

### Price Discrimination and Elasticity
**General Rule for N Markets**
P_i[1 - 1/ε_i] = C for all i = 1,...,N

**For Two Markets**
P_1/P_2 = (1 - 1/ε_2)/(1 - 1/ε_1)

**Key Insight**: Elasticities determine relative prices
- When ε_2 rises, P_1/P_2 rises as well
- Price ratio depends on elasticity ratio

### Product Differentiation
**Definition** (Phlips): Price discrimination exists when "two varieties of a commodity are sold by the same seller to two buyers at different net prices, the net price being the price paid corrected for cost associated with product differentiation"

**Examples**
- Haircuts (male vs. female)
- Academic licenses
- Business vs. economy airfare

**Key Finding**: Even with identical costs or identical demands, price differences typically won't equal cost differences
- With identical demand P_i = A - BQ_i and different costs c_j = c_i + t:
- Price difference: P_j - P_i = (A_i - A_j)/2 + (c_i - c_j)/2
- Unlikely that price difference = cost difference

### Discrimination by Location
**Setup**: Identical demands, different supply costs
- Example: Two amusement parks with similar demographics but different labor costs

**Result**: P_j - P_i = t/2 ≠ c_j - c_i
- Price difference is NOT the same as cost difference even with identical demand!

### Other Discrimination Mechanisms
**Usage Restrictions** (to prevent arbitrage):
- Saturday night stay requirements
- No changes/alterations policies
- Personal use only restrictions
- Time-based restrictions (movies, restaurants)

**Product "Crimping"**:
- Deliberately degrade products for lower-tier markets
- Example: Mathematica® versions

### Welfare Effects
**Main Principle**: Price discrimination cannot increase surplus unless it increases aggregate output

**Standard Case** (Same markets served):
- Change in welfare ΔW ≤ (P_U - MC)(ΔQ_1 + ΔQ_2)
- Maximum gain in weak market: G
- Minimum loss in strong market: L
- ΔW ≤ G - L

**New Markets Case**:
- When uniform pricing serves only "strong" markets
- Price discrimination may open "weak" markets
- If MC < reservation price in weak market, discrimination increases welfare
- Creates both consumer surplus and profit in new market
- Example: Lower vaccination prices in developing countries

**Conclusion**: Price discrimination can increase welfare when it:
1. Increases aggregate output
2. Opens new markets that wouldn't be served otherwise

---

## L9: Second Degree Price Discrimination

### Introduction - Degrees of Price Discrimination

**First-Degree Price Discrimination**
- Firm knows different consumers have different demand functions
- Firm can identify who is who
- Extracts all consumer surplus, usually with a two-part tariff

**Second-Degree Price Discrimination** (Focus of this lecture)
- Firm knows different consumers have different demand functions
- Firm CANNOT tell who is who
- Firm offers a menu of different packages/options
- Designed for consumers to self-select
- Unable to extract all consumer surplus

### Key Characteristics of Second-Degree Discrimination

**The Information Problem**
- Firm knows consumers have different willingness to pay (WTP)
- Cannot identify individual consumer types
- Examples:
  - Higher income people willing to pay more, but cannot tell people's income
  - Business travelers less flexible and willing to pay more, but cannot identify them

**Solution Mechanism**
- Offer different packages allowing consumers to self-select into different price categories

### Theoretical Framework

**Model Setup**
- N identical high-income consumers with inverse demand: P_H = A - Q
- n identical low-income consumers with inverse demand: P_L = a - Q
- A > a > 0 (high-income consumers have higher WTP)
- Cost function: C = cQ with 0 < c < a

**Willingness to Pay Functions**
- High-income WTP: W_H(Q) = ∫₀^Q P_H(x)dx = AQ - Q²/2
- Low-income WTP: W_L(Q) = ∫₀^Q P_L(x)dx = aQ - Q²/2

### Three Strategic Options

**Option 1: Target Only High-Income Consumers**
- Constraints: V > W_L(Q) but V ≤ W_H(Q)
- Optimal: Charge V = W_H(Q)
- Solution: Q*₁ = A - c
- Price: V*₁ = (A² - c²)/2
- Profit: π*₁ = N(A - c)²/2

**Option 2: Target Both Consumer Types with One Package**
- Constraints: V ≤ W_L(Q) (automatically satisfies V < W_H(Q))
- Optimal: Charge V = W_L(Q)
- Solution: Q*₂ = a - c
- Price: V*₂ = (a² - c²)/2
- Profit: π*₂ = (N + n)(a - c)²/2

**Option 3: Offer Two Different Packages**

Four constraints must be satisfied:
1. V_L ≤ W_L(Q_L) - L-consumers willing to buy their package
2. W_L(Q_L) - V_L ≥ W_L(Q_H) - V_H - Incentive compatibility for L-consumers
3. V_H ≤ W_H(Q_H) - H-consumers willing to buy their package
4. W_H(Q_H) - V_H ≥ W_H(Q_L) - V_L - Incentive compatibility for H-consumers

**Key Insight from Constraints**
- Constraint (3) follows from (1) and (4)
- From incentive compatibility for H-consumers: H-consumers MUST get positive surplus
- Price of H-package < their willingness to pay

### Optimal Two-Package Solution

**Binding Constraints**
- (1) and (4) must be satisfied as equalities:
  - V_L = W_L(Q_L)
  - V_H = W_H(Q_H) - W_H(Q_L) + W_L(Q_L) = AQ_H - (Q_H)²/2 - (A - a)Q_L

**Profit Maximization**
- π = N(V_H - cQ_H) + n(V_L - cQ_L)
- Solving first-order conditions:
  - Q*_H = A - c
  - Q*_L = a - c - (N/n)(A - a)

**Feasibility Condition**
- Solution is acceptable iff Q*_L > 0
- This requires: n(a - c) > N(A - a)

**Verification of Constraint (2)**
- Since (1) satisfied as equality, LHS of (2) = 0
- RHS of (2) = W_L(Q*_H) - V*_H = -(A - a)(Q*_H - Q*_L) < 0
- Therefore constraint (2) is automatically satisfied

**Maximum Profit**
- π*₃ = (N + n)(a - c)²/2 + (N + n)N(A - a)²/(2n)
- π*₃ = π*₂ + (N + n)N(A - a)²/(2n)
- Option 2 is ALWAYS inferior to Option 3

### Decision Rule

**Comparing Options 1 and 3**
- Use Option 3 (two packages) if: n(a - c) > N(A - a)
- Use Option 1 (only high-income) if: n(a - c) < N(A - a)

**Economic Intuition**
- Marginal WTP of L-consumer is a
- Introducing L-package increases profit from L-consumers by n(a - c)
- But this gives H-consumers surplus of (A - a)
- Must reduce H-package price by (A - a), losing revenue of N(A - a)
- Profitable to introduce L-package only if gain > loss

### Practical Examples

**Example 1: Option 1 Optimal**
- N = n, A = 10, a = 4, c = 2
- n(a - c) = 2n < N(A - a) = 4n
- Monopolist chooses Option 1
- π*₁ = 32n, π*₂ = 4n (π*₃ not defined as Q*_L = -4)

**Example 2: Option 3 Optimal**
- N = n, A = 10, a = 8, c = 2
- n(a - c) = 6n > N(A - a) = 2n
- Monopolist chooses Option 3
- π*₁ = 32n, π*₂ = 36n, π*₃ = 40n
- Solution: Q*_H = 8, Q*_L = 4, V*_H = 40, V*_L = 24
- Effective per-unit prices: P_H = 5, P_L = 6
- H-package incorporates a **quantity discount**

### Key Takeaways

1. **Self-Selection Mechanism**: When firms cannot identify consumer types, they design menus that induce consumers to reveal their preferences through choices

2. **Information Rent**: High-WTP consumers must receive positive surplus (information rent) to prevent them from choosing the low-price package

3. **Distortion at the Bottom**: Low-WTP consumers get less than efficient quantity (Q*_L < a - c) to make their package unattractive to high-WTP consumers

4. **No Distortion at the Top**: High-WTP consumers get efficient quantity (Q*_H = A - c)

5. **Trade-off**: Firm balances between serving low-WTP market vs. extracting more surplus from high-WTP consumers

6. **Quantity Discounts**: Often implements as quantity discounts where per-unit price decreases with quantity purchased

---

## L8: Second Degree Price Discrimination (continued) and Welfare Analysis

### Should Monopolist Serve All Consumer Types?

**Strategic Decision**
- Not always optimal to serve both consumer types
- Sometimes better to serve only high-demand consumers
- Examples in practice:
  - High-class restaurants
  - Golf and country clubs

### Numerical Example: Choosing Which Markets to Serve

**Setup**
- N_l low-income consumers
- N_h high-income consumers

**Option A: Serve Both Types**
- Packages: ($57.50, 7) for low-income, ($92, 12) for high-income
- Profit: π = $31.50 × N_l + $44 × N_h

**Option B: Serve Only High-Income**
- Package: ($120, 12)
- Profit: π = $72 × N_h

**Profitability Condition**
- Serve both types only if: $31.50 × N_l + $44 × N_h > $72 × N_h
- Simplifies to: 31.50N_l > 28N_h
- Rearranging: N_h/N_l < 31.50/28 = 1.125

**Key Insight**: Should NOT serve both types if there is "too high" a fraction of high-demand consumers

### Characteristics of Second-Degree Price Discrimination

**Surplus Extraction**
1. Extract ALL consumer surplus from lowest-demand group
2. Leave SOME consumer surplus for other groups (due to incentive compatibility constraint)

**Quantity Distortion**
- Offer less than socially efficient quantity to all groups EXCEPT highest-demand group
- Only highest-demand group gets efficient quantity

**Pricing Structure**
- Typically implemented through quantity discounting
- Per-unit price decreases with quantity purchased

**Comparison to First-Degree**
- Less effective at converting consumer surplus into profit than first-degree discrimination
- Some consumer surplus left "on the table" to induce high-demand groups to buy large quantities

### Welfare Analysis of Non-Linear Pricing

**General Framework**
- Inverse demand of consumer group i: P = P_i(Q)
- Marginal cost: constant at MC = c
- Quantity offered to group i: Q_i
- Total surplus = Consumer surplus + Profit
- Total surplus = area between inverse demand curve and MC up to quantity Q_i

**Two Effects of Pricing Policy**
1. **Distribution of surplus** (welfare neutral)
2. **Output of the firm** (affects welfare)

**Welfare Impact Rule**
Price discrimination increases social welfare of group i IF AND ONLY IF it increases quantity supplied to group i

### First-Degree Price Discrimination and Welfare

**Welfare Properties**
- ALWAYS increases social welfare
- Extracts all consumer surplus
- But generates socially optimal output
- Output to group i is Q_i(c) (where P_i(Q_i) = c)
- This exceeds output with uniform (non-discriminatory) pricing

**Conclusion**: First-degree discrimination achieves allocative efficiency despite distributional concerns

### Second-Degree (Menu) Pricing and Welfare

**Two-Market Analysis**

Market structure:
- Low-demand market
- High-demand market
- Uniform price: P_U
- Menu pricing quantities: Q_ls (low), Q_hs (high)
- Uniform pricing quantities: Q_lU (low), Q_hU (high)

**Key Observations**
1. High-demand group: offered socially optimal quantity
2. Low-demand group: offered LESS than socially optimal quantity

**Welfare Calculation**
- Welfare loss in low-demand market > L (deadweight loss from reduced quantity)
- Welfare gain in high-demand market < G (gain from increased quantity)
- Net welfare change: ΔW < G - L

**Mathematical Expression**
- ΔW = (P_U - MC)ΔQ_1 + (P_U - MC)ΔQ_2
- ΔW = (P_U - MC)(ΔQ_1 + ΔQ_2)

**Necessary Condition for Welfare Improvement**
Second-degree price discrimination increases social welfare ONLY IF it increases total output (ΔQ_1 + ΔQ_2 > 0)

### Comparison with Third-Degree Price Discrimination

**Similarities**
- Both require total output increase for welfare improvement
- Same basic welfare condition

**Key Difference**
- Second-degree price discrimination is MORE LIKELY to increase output than third-degree
- Therefore more likely to improve social welfare

### The Incentive Compatibility Constraint - Broader Applications

**Core Principle**
Any offer made to high-demand consumers must offer them at least as much consumer surplus as they would get from an offer designed for low-demand consumers

**Applications Beyond Pricing**

1. **Performance Bonuses**
   - Must encourage effort
   - Workers must prefer working hard with bonus over shirking

2. **Insurance Policies**
   - Need large deductibles to deter cheating
   - High-risk customers must not prefer low-risk policies

3. **Factory Piece Rates**
   - Must be accompanied by strict quality inspection
   - Workers must not sacrifice quality for quantity

4. **Bulk Purchase Discounts**
   - Must offer sufficient price discount
   - High-volume buyers must prefer bulk package over multiple small purchases

### Summary of Key Welfare Results

**First-Degree Discrimination**
- Allocatively efficient (socially optimal output)
- All consumer surplus extracted
- Always welfare improving compared to monopoly pricing

**Second-Degree Discrimination**
- NOT allocatively efficient
- Only highest-demand group gets efficient quantity
- All other groups face quantity distortion (Q < Q*)
- Welfare improvement requires total output increase
- More likely to improve welfare than third-degree discrimination

**Policy Implications**
- Cannot judge price discrimination on distributional grounds alone
- Must consider impact on total output and allocative efficiency
- Second-degree discrimination represents a middle ground between efficiency and surplus extraction

