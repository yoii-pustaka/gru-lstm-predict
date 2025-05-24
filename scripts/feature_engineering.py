def create_features(df):
    df = df.copy()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    # tambah fitur lain sesuai kebutuhan
    df = df.dropna()
    return df
