
# Part 1: Euclidean Distance
Why is Euclidean distance appropriate for this dataset?
- Euclidean is natural for continuous feature space like Iris (flower dimensions).

How would changing k affect your accuracy?
- Small k → more sensitive to noise (overfitting). Large k → smoother, risk of underfitting.

# Part 2: Manhattan Distance
Why is Manhattan distance more suitable here?
- Grid-like dataset = blocky structure → Manhattan (city-block distance) is more appropriate.

What happens if you use Euclidean distance instead?
- If you use Euclidean, the circular neighborhoods may cut through irrelevant grid cells, lowering accuracy.

# Part 3: Decision Boundaries
How does the choice of distance metric affect the shape of the boundary?
- Euclidean → smooth, circular/elliptical boundaries.
- Manhattan → square-like, aligned with axes (grid-structured).

Can you explain why it looks the way it does?
- The shape follows how distance is computed: straight-line vs city-block.

# Part 4: Experimenting with K
Which k gives the best performance?
- Best k usually small but not too small (often 3–7).

How does a very small k vs very large k affect overfitting/underfitting?
- Very small k (e.g., 1) → overfitting, jagged boundaries.
- Very large k → oversmoothing, can misclassify minority regions.