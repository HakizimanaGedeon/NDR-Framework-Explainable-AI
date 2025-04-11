from ndr.framework import NDRFramework

# Initialize NDR framework
ndr = NDRFramework()

# Add laws and antecedents
ndr.add_law("Law1", "If Mean Radius is high, the likelihood of cancer increases.")
ndr.add_antecedent("Antecedent1", "Mean Radius > 15")

# Generate explanation
features = {'mean_radius': 16, 'smoothness': 0.2, 'mean_texture': 22}
explanation = ndr.generate_explanation(features)
print(explanation)
