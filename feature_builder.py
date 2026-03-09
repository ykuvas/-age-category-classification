
import pandas as pd

def build_features(users, visits, ads_activity, surf_depth, primary_device, cloud_usage):
    users = users.drop_duplicates("user_id").copy()
    v = visits.drop_duplicates().copy()

    visits_agg = (v.groupby("user_id")
        .agg(
            sessions_total=("session_id", "nunique"),
            active_days=("date", "nunique"),
        )
        .reset_index()
    )

    daytime_share = (
        pd.crosstab(v["user_id"], v["daytime"], normalize="index")
          .add_prefix("daytime_share_")
          .reset_index()
    )

    cat_share = (
        pd.crosstab(v["user_id"], v["website_category"], normalize="index")
          .add_prefix("cat_share_")
          .reset_index()
    )

    visits_features = (visits_agg
        .merge(daytime_share, on="user_id", how="left")
        .merge(cat_share, on="user_id", how="left")
    )

    ads_activity = ads_activity.drop_duplicates("user_id")
    surf_depth = surf_depth.drop_duplicates("user_id")
    primary_device = primary_device.drop_duplicates("user_id")
    cloud_usage = cloud_usage.drop_duplicates("user_id")

    df = (users
        .merge(visits_features, on="user_id", how="left")
        .merge(ads_activity, on="user_id", how="left")
        .merge(surf_depth, on="user_id", how="left")
        .merge(primary_device, on="user_id", how="left")
        .merge(cloud_usage, on="user_id", how="left")
    )

    user_id = df["user_id"].copy() if "user_id" in df.columns else None
    y = df["age_category"].copy() if "age_category" in df.columns else None
    X = df.drop(columns=[c for c in ["user_id", "age_category"] if c in df.columns])

    return X, y, user_id
