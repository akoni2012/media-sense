import pandas as pd

# Standardize the values (based on their individual benchmarks) of the features in order to make them relatable
# achieved by subtracting the benchmark from the grand total of each dimension (e.g. viewability dimension) and 
# then dividing the result by the benchmark. Multiply by -100 or +100 in order to standardise to whole +ive numbers
# which are meaningful as ratings. Returns a list of ratings for all dimensions.
# Rating - Media-sense universal measure of how good or bad a campaign fared.
def generate_ratings(path_to_file):

    # read the contents of the performance metrics into a Pandas data frame
    df = pd.read_csv(path_to_file)

    features = ['Viewability', 'Brand Safety Risk', 'IVT', 'Out-of-Geo']

    # convert percentages (str) into numbers/decimals for processing
    for i in features:
        replace_with_floats = df[i].str.rstrip("%").astype(float)/100
        df[i] = replace_with_floats    

    viewability = 100*(df.loc[5]['Viewability'] - df.loc[6]['Viewability'])/df.loc[6]['Viewability']
    brand_safety_risk = -100*(df.loc[5]['Brand Safety Risk'] - df.loc[6]['Brand Safety Risk'])/df.loc[6]['Brand Safety Risk']
    invalid_traffic = -100*(df.loc[5]['IVT'] - df.loc[6]['IVT'])/df.loc[6]['IVT']
    out_of_geo = -100*(df.loc[5]['Out-of-Geo'] - df.loc[6]['Out-of-Geo'])/df.loc[6]['Out-of-Geo']

    ratings = {'viewability': round(viewability), 'brand_safety_risk': round(brand_safety_risk), 'invalid_traffic': round(invalid_traffic), 'out_of_geo': round(out_of_geo)}
    return ratings

# ratings = generate_ratings()
# print(ratings)