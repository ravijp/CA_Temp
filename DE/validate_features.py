import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.types import *

# from IPython.display import display


class FeatureValidator:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.validators = [
            {"name": "duplicate_rows", "fn": self.has_duplicate_rows},
            {"name": "duplicate_persons", "fn": self.has_duplicate_persons},
        ]

    def run_validations(self, df: DataFrame):
        all_pass = True
        for v in self.validators:
            vname, vfunc = v["name"], v["fn"]
            if vfunc and callable(vfunc):
                try:
                    vres = vfunc(df)
                except Exception as e:
                    print(f"Error invoking validator: {vname}: {e}")
                    vres = False
                    continue
            else:
                print(f"Function for validator {vname} is either NoneType or not callable: {vfunc}")
                vres = False
                continue
            if vres:
                print(f"Validation {vname} successful")
            else:
                print(f"Validation {vname} failed")
            
            all_pass = all_pass and vres
        
        if all_pass:
            print("All validations passed!")
        else:
            print("One or more validations have failed. Check above messages for details")

    def has_duplicate_rows(self, df: DataFrame) -> bool:
        orig_count = df.count()
        dedup_count = df.dropDuplicates().count()
        if orig_count != dedup_count:
            print(f"Original count: {orig_count}, After dropDuplicates: {dedup_count}")
            return False
        else:
            print("No duplicate rows")
            return True

    def has_duplicate_persons(self, df: DataFrame) -> bool:
        orig_count = df.count()
        dedup_count = df.dropDuplicates(["person_composite_id", "vantage_date"]).count()
        if orig_count != dedup_count:
            print(
                f"Original count: {orig_count}, After dropDuplicates on person_composite_id and vantage_date: {dedup_count}"
            )
            return False
        else:
            print("No duplicate persons on person_composite_id and vantage_date")
            return True