## 1. Setup & Planning
 Set up a shared GitHub repo and split tasks (e.g., one person does classification tree + gain ratio, the other does regression tree + preprocessing; split pruning and the CV harness after that)
 
 Agree on a shared data representation that tracks feature type (numeric vs. categorical) per column, since the tree needs to know which is which
 
 Agree on a common interface (e.g., fit(X, y, feature_types), predict(X), prune(X_prune, y_prune))
 
 Decide on a tree node structure (feature, threshold or child dictionary, majority class or mean value, sample count)
## 2. Data Acquisition & Preprocessing

Get one dataset working end to end before scaling to all six.

 Download all 6 datasets (UCI or Canvas) and read each .names file
 
 Remove features that act as unique identifiers:
 -Breast Cancer: Sample code number
-Computer Hardware: vendor and model name (drop ERP as well, since it is the authors' linear regression estimate, not a real feature; PRP is the target)

 Handle missing values:
-Breast Cancer: ? in Bare Nuclei (impute or drop; document the choice)
-Congressional Vote: ? means abstain, so keep it as its own categorical value

 Label each column as numeric or categorical (nominal/ordinal). Decide and document how you treat Breast Cancer's 1–10 features (numeric vs. ordinal)
 
 Do not one-hot encode. Abalone's sex (M/F/I), Forest Fires' month/day, and Car Evaluation's features stay categorical and get multiway splits
 
 Forest Fires: decide whether to log-transform area (it is heavily skewed); document your choice
 
 Skip normalization (axis-parallel splits don't need it)
 
 Produce a clean, consistent structure for all 6 datasets
## 3. Core Building Blocks
 Entropy function H(D)
 
 Information gain
 
 Intrinsic value IV(f) and gain ratio (guard against IV = 0)
 
 Candidate threshold generation for numeric features (midpoints between sorted unique values) for binary splits
 
 Multiway split logic for categorical features (one child per discrete value)
 
 Weighted MSE split criterion for regression (the Err′ formula in the spec)
 
 Evaluation metrics: classification accuracy or 0/1 loss, and mean squared error
 
## 4. Null Models (Baselines)
 Classification: predict the plurality class from training data
 
 Regression: predict the mean of training responses
## 5. Classification Tree (Gain Ratio)
 Recursive tree builder that picks the feature/threshold with the highest gain ratio
 
 Compare numeric binary splits and categorical multiway splits under the same criterion, and document how you do it
 
 Stopping conditions for growing to completion (node is pure, no features left, or no samples)
 
 Handle unseen categorical values at prediction time (e.g., fall back to the node's majority class)
 
 Predict function that traverses the tree
 
 Sanity check on one dataset: unpruned tree should reach ~100% training accuracy and beat the null model on test data
## 6. Regression Tree (MSE)
 Recursive builder that picks the split minimizing weighted MSE
 
 Leaf prediction is the mean response of the samples in the node
 
 Stopping conditions for growing to completion (e.g., zero variance or a single sample)
 
 Same handling of numeric vs. categorical splits and unseen values
 
 Sanity check on one dataset against the null model
## 7. Reduced Error Pruning (Post-Pruning Only)
 Implement bottom-up pruning: for each internal node, compare the pruning-set error with the subtree vs. with the node replaced by a leaf
 
 Prune when the leaf's error is less than or equal to the subtree's error (document your tie-break rule)
 
 Make sure leaf predictions (majority class or mean) come from training data at that node, and only the pruning set decides whether to prune
 
 Repeat until no more pruning improves the error
 
 Track tree size (node/leaf count) before and after pruning
 
 Confirm you are not using pre-pruning or early stopping
## 8. Data Splitting & 5×2 Cross-Validation Harness
 Pull out a 20% pruning set first (stratified for classification) and keep it fixed
 
 Run 5×2 CV on the remaining 80%: 5 repetitions of a 2-fold split with shuffling
 
 For each fold: train an unpruned tree on the training fold, prune a copy using the 20% pruning set, then test both on the held-out fold
 
 Run the null model on the same folds
 
 Run all methods (null, unpruned, pruned) across all 6 datasets
 
 Save raw predictions, scores, and tree sizes to files rather than only printing them
 
 Set random seeds and document them for reproducibility
## 9. Results Analysis
 Aggregate results into tables (mean ± std for accuracy or MSE per dataset and method)
 
 Add a table of tree size before vs. after pruning
 
 Statistical comparison of pruned vs. unpruned vs. null (e.g., paired t-test, Wilcoxon signed-rank, or the 5×2 cv F-test), and state your significance level
 
 Figures such as performance by method per dataset, tree size before vs. after pruning, and a boxplot of fold scores
 
 Optionally print a few learned rules from the trees and check whether they "make sense," as the spec suggests
 
 Note where pruning helped, hurt, or made no difference, and think about why (dataset size, noise, feature types)
## 10. Report Writing (JMLR format, ≤15 pages + appendices)
 Title and authors
 
 Problem statement with at least one testable hypothesis (e.g., "Reduced error pruning will improve test accuracy or MSE relative to a fully grown tree on noisy datasets, and will produce significantly smaller trees")
 
 Experimental approach and project design (datasets, preprocessing choices, 80/20 pruning split, 5×2 CV)
 
 Algorithm explanations without code (gain ratio, binary numeric splits, multiway categorical splits, MSE splitting, reduced error pruning)
 
 Results (tables and figures with explanations)
 
 Analysis and discussion, tied back to your hypothesis
 
 Conclusion
 
 References (if using any beyond the course content)
 
 Appendix A: main lessons learned (not counted in the page limit)
 
 Appendix B: who did what (not counted in the page limit)
## 11. Final Checks
 All math is typeset in LaTeX (including the gain ratio, IV, and MSE formulas)
 
 Figures are vector or high quality, with no screenshots
 
 Proofread against JMLR format requirements
 
 Confirm the 15-page limit (excluding appendices)
 
 Export to PDF and check plagiarism-safe wording (the report goes through TurnItIn)
 
 Submit the PDF to "P2 Paper" on Canvas (the old checklist said P1)
