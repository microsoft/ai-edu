import RentCar_0_Data as carData
import RentCar_1_FromB as car_1

if __name__ == "__main__":
    # 读取文件
    data_array = carData.read_data()
    for rent_from in car_1.RentalStore:
        print(str.format("从 {0} 店租出：", rent_from.name))
        for return_to in car_1.RentalStore:
            num_from, num_to = car_1.Statistic(data_array, rent_from, return_to, t=1)
            print(str.format(
                "还到 {0} 店：租出次数={1}, \t归还次数={2}, \tP({4}|{5})={3}", 
                return_to.name, num_from, num_to, num_to/num_from, return_to.name, rent_from.name))
