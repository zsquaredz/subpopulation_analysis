from create_dataset import ReviewDataProvider
import utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data")
    parser.add_argument("--category", type=str, default='Books', help="category")
    parser.add_argument("--data_dir", type=str, default='./data/', help="Directory of data")
    parser.add_argument("--reviews_per_class", type=int, default=1000, help="Number of reviews per class")
    parser.add_argument("--train_size", type=int, default=1000, help="Training size desired to get")
    parser.add_argument("--val_size", type=int, default=1000, help="Val size desired to get")
    parser.add_argument("--test_size", type=int, default=1000, help="Test size desired to get")
    parser.add_argument("--just_create_split", action='store_true', help="whether to just create split")
    args = parser.parse_args()
    categories = ["Books", "Electronics", "Movies_and_TV", "CDs_and_Vinyl", "Clothing_Shoes_and_Jewelry",
                  "Home_and_Kitchen", "Kindle_Store", "Sports_and_Outdoors", "Cell_Phones_and_Accessories",
                  "Toys_and_Games", "Video_Games", "Tools_and_Home_Improvement",
                  "Office_Products", "Pet_Supplies", "Automotive",
                  "Grocery_and_Gourmet_Food", "Patio_Lawn_and_Garden", "Arts_Crafts_and_Sewing", "Musical_Instruments"]
    
    # above categories (19 domains) are taken from https://nijianmo.github.io/amazon/index.html : "Small" subsets for experimentation
    # excluding: Amazon Fashion, All Beauty, Appliances, Digital_Music, Gift Cards, Industrial and Scientific, Luxury Beauty,
    # Magazine Subscriptions, Prime_Pantry, Software
    
    assert args.category in categories
    rdp = ReviewDataProvider(args.data_dir, args.category)
    if not args.just_create_split:
        print('constructing dataset from gz files')
        reviews, labels = rdp.construct_dataset(args.reviews_per_class)
    
    X_train, y_train, X_val, y_val, X_test, y_test = utils.load_existing_dataset_and_create_splits(
        args.data_dir, 
        args.category,
        args.train_size,
        args.val_size,
        args.test_size)
    rdp.create_json(X_train, y_train, 'Train_'+str(args.train_size)+'_'+args.category+'.json')
    rdp.create_txt(X_train, y_train, 'Train_'+str(args.train_size)+'_'+args.category+'.txt', 
                    'Train_'+str(args.train_size)+'_'+args.category+'_label.txt')
    rdp.create_json(X_val, y_val, 'Val_'+str(args.val_size)+'_'+args.category+'.json')
    rdp.create_txt(X_val, y_val, 'Val_'+str(args.val_size)+'_'+args.category+'.txt', 
                    'Val_'+str(args.val_size)+'_'+args.category+'_label.txt')
    rdp.create_json(X_test, y_test, 'Test_'+str(args.test_size)+'_'+args.category+'.json')
    rdp.create_txt(X_test, y_test, 'Test_'+str(args.test_size)+'_'+args.category+'.txt', 
                    'Test_'+str(args.test_size)+'_'+args.category+'_label.txt')
   



