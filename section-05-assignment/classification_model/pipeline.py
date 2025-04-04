from feature_engine.imputation import CategoricalImputer, AddMissingIndicator, MeanMedianImputer
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from feature_engine.selection import DropFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from classification_model.config.core import config
from classification_model.processing import features as pp

titanic_pipe = Pipeline([

    # ===== PREPROCESSING =====
    # Retain only the first cabin
    ('get_first_cabin', pp.GetFirstCabinTransformer()),
    # Extract the title from the name
    ('get_title', pp.GetTitleTransformer()),
    # Cast numerical variables to float
    ('cast_num', pp.CastNumericalTransformer(config.model_config.numerical_variables)),
    # ===== IMPUTATION =====
    # impute categorical variables with string missing
    ('categorical_imputation', CategoricalImputer(
        imputation_method='missing', variables=config.model_config.categorical_variables)),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(variables=config.model_config.numerical_variables)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(
        imputation_method='median', variables=config.model_config.numerical_variables)),


    # Extract letter from cabin
    ('extract_letter', pp.ExtractLetterTransformer(variables=config.model_config.extract_letter_variables)),

    # ===== FEATURE SELECTION =====
    # drop features
    ('drop_features', DropFeatures(features_to_drop=config.model_config.drop_variables)),

    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(
        tol=config.model_config.rare_label_tolerance,
        n_categories=config.model_config.rare_label_n_categories,
        variables=config.model_config.categorical_variables)),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(
        drop_last=True, variables=config.model_config.categorical_variables)),

    # scale
    ('scaler', StandardScaler()),

    ('Logit', LogisticRegression(C=config.model_config.regression_c, random_state=config.model_config.random_state)),
])