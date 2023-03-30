from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Helper:
    def train_val_test_split(x, y, split_size, rand_state):

        # after researching a bit, it appears as though the common practice for a train val test split
        # is to use the follow equation: N / (1 - N)
        # where N = the split ratio. This ensures that your val and test portions are equally sized.
        
        val_split_size = split_size / (1 - split_size)

        # Train val test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=split_size, random_state=rand_state
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=val_split_size, random_state=rand_state
        )
        
        return (x_train, x_val, x_test, y_train, y_val, y_test)
    
    
    def scaler(type: str, x1, x2):
        if type == "standard":
            scaler = StandardScaler() # should be class variable?
        else:
            scaler = MinMaxScaler()

        x1 = scaler.fit_transform(x1)
        x2 = scaler.transform(x2)

        return (x1, x2)
    