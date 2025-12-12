
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class MarketBasketAnalysis:
    def __init__(self):
        self.rules = None
        self.frequent_itemsets = None

    def prepare_basket(self, df):
        """Converts raw transaction df to basket format (Invoice x Description)."""
        # Filter for France or just take small sample to speed up, otherwise it's very slow
        # For demonstration we use the whole df but aggregate carefully
        basket = (df.groupby(['Invoice', 'Description'])['Quantity']
                  .sum().unstack().reset_index().fillna(0)
                  .set_index('Invoice'))
        
        # Convert to boolean (1/0)
        basket_sets = basket.map(lambda x: 1 if x >= 1 else 0)
        return basket_sets

    def run_apriori(self, basket_sets, min_support=0.05, min_confidence=0.1):
        self.frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
        self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        return self.rules
