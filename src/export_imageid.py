import pandas as pd

def extract_unique_images(csv_path):
    """
    Reads a CSV of menu variations and returns a DataFrame 
    of unique image_ids with a single representative name.
    """
    # 1. Read the CSV
    df = pd.read_csv(csv_path)

    # 2. Drop duplicates based on image_id
    unique_images = df.drop_duplicates(subset=['image_id'], keep='first')

    # 3. Keep required columns
    result = unique_images[['image_id', 'menu_item_name']].rename(
        columns={'menu_item_name': 'name'}
    )

    # 4. Sort for consistency
    result = result.sort_values('image_id').reset_index(drop=True)

    return result


if __name__ == "__main__":
    # 🔹 Load real CSV file
    unique_db = extract_unique_images("data.csv")

    print("✅ Unique Image Master List:")
    print(unique_db)

    # 🔹 Save cleaned master list
    unique_db.to_csv("master_image_list.csv", index=False)
