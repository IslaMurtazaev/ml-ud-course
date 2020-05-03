#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    errors = (net_worths - predictions) ** 2
    cleaned_data = zip(ages, net_worths, errors)
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2][0])
    limit = len(predictions) - len(predictions) / 10
    print 'cleaned data', cleaned_data

    return cleaned_data[:limit]

