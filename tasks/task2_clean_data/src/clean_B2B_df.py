import os 
import pandas as pd
import networkx as nx

def filter_giant_component(B2B_df):
    frames = []
    for year, df_year in B2B_df.groupby('year'):
        G = nx.from_pandas_edgelist(
            df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph()
        )
        if G.number_of_nodes() == 0:
            continue

        largest_wcc = max(nx.weakly_connected_components(G), key=len)

        n_nodes_before = G.number_of_nodes()
        n_edges_before = G.number_of_edges()
        n_nodes_after = len(largest_wcc)
        df_filtered = df_year[
            df_year['vat_i'].isin(largest_wcc) & df_year['vat_j'].isin(largest_wcc)
        ]
        n_edges_after = len(df_filtered)

        print(
            f"Year {year}: "
            f"nodes {n_nodes_before} → {n_nodes_after} "
            f"(dropped {n_nodes_before - n_nodes_after}, "
            f"{(n_nodes_before - n_nodes_after) / n_nodes_before * 100:.2f}%), "
            f"edges {n_edges_before} → {n_edges_after} "
            f"(dropped {n_edges_before - n_edges_after}, "
            f"{(n_edges_before - n_edges_after) / n_edges_before * 100:.2f}%)"
        )

        frames.append(df_filtered)

    return pd.concat(frames, ignore_index=True) if frames else B2B_df.iloc[0:0]


# Master function
def clean_B2B_df(tmp_path, B2B_df):
    
        B2B_df = B2B_df[B2B_df['sales_ij'] != 0]
        
        #B2B_df = filter_giant_component(B2B_df)
    
        B2B_df.to_parquet(os.path.join(tmp_path, 'B2B_data_cleaned.parquet'), engine='pyarrow')

