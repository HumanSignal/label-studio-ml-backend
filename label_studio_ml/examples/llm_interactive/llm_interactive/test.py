import re
import uuid


def match_errors(self, original_text, error_text):
    def find_error_position(error_text, source_text):
        start = source_text.find(error_text)
        end = start + len(error_text)
        return start, end

    # Функция для генерации уникального идентификатора
    def generate_id():
        return str(uuid.uuid4())

    # Разбор текста с описаниями ошибок
    error_descriptions = {
        "Fluency errors": [],
        "Grammatical errors": [],
        "Additional input values": [],
        "Missing input values": [],
        "Repetitions": []
    }

    # Изменение регулярного выражения для извлечения текста ошибок
    error_regex = r'<"([^">]+)"'

    # Обработка текста с ошибками
    for category in error_descriptions.keys():
        category_regex = f"{category}:(.*?)(\n#|\Z)"
        category_text = re.search(category_regex, error_text, re.DOTALL)
        if category_text:
            errors = re.findall(error_regex, category_text.group(1), re.DOTALL)
            error_descriptions[category].extend(errors)

    # Поиск позиций ошибок в исходном тексте и формирование JSON
    json_output = []
    for category, errors in error_descriptions.items():
        for error in errors:
            start, end = find_error_position(error, original_text)
            if start != -1:
                json_output.append({
                    "from_name": "errors",
                    "id": generate_id(),
                    "to_name": "Generation",
                    "type": "labels",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": error,
                        "labels": [category]
                    }
                })

    return json_output


original_text = """
Experience warm hospitality, clean, affordable accommodations in the 4 star hotel Hotel Torre Azul, that is conveniently located in heart of El Arenal .The hotel amenities include a entrance hall, elevator, luggage storage service and complimentary wifi access.The hotel also features a locker service.The reception is available for any enquiries throughout the day. Check-in/Check-out can be done by without attending the reception. Cleaning service is also available.The hotel offers a complimentary breakfast.There is a public bar available in the hotel with regular opening hours.A cafe on-site provides light drinks and snacks.The in-house restaurant offers delicious meals.The management organizes activities inorder keep the guests engaged.It also has its own arcade/video games.Amenities intended to make the stay even better beyond expectation include an outdoor pool.The guests can relax in the pool along with some drinks and snacks from the pool bar.The hotel features a sauna, massage, hammam, spa, whirlpool, jacuzzi, steam room and beauty center.The guests can enjoy a relaxing stay.All the rooms are centrally heated.Each room has its own wifi-connection.The rooms have a balcony that are private to the guests.The guests can access Web within the room via broadband or Wi-Fi.The living space feature a satellite TV and a television.
"""
errors = """
# Fluency errors:
No errors detected.

# Grammatical errors:
<"Check-in/Check-out can be done by without attending the reception.">
Error clarification: The sentence is grammatically incorrect. It should be "Check-in/Check-out can be done without attending the reception."

# Additional input values:
No errors detected.

# Missing input values:
No errors detected.

# Repetitions:
<"The guests can access Web within the room via broadband or Wi-Fi.">
<"Each room has its own wifi-connection.">
Error clarification: There is a repetition in mentioning the Wi-Fi availability in the rooms.

<"The living space feature a satellite TV and a television.">
Error clarification: The text repeats the information about television availability in the rooms.

<"The guests can enjoy a relaxing stay.">
<"The guests can relax in the pool along with some drinks and snacks from the pool bar.">
Error clarification: The text repeats the information about relaxation options for guests.
"""

print(match_errors(None, original_text, errors))
