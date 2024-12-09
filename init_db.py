import os
import psycopg2
from PIL import Image, ImageDraw, ImageFont
from flask_bcrypt import Bcrypt
import bcrypt
import random
import string

# Path to save generated images
output_dir = "/home/sgeede/Documents/generated_images/"

# Create directory if it doesn't exist
import os
os.makedirs(output_dir, exist_ok=True)

conn = psycopg2.connect(
        host="localhost",
        database="ai_recog_db",
        user="postgres",
        password="postgres")

username = 'admin'  # Change as needed
password = 'securepassword'  # Change as needed

# Hash the password
salt = bcrypt.gensalt()  # Generates a salt
hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a command: this creates a new table
cur.execute('DROP TABLE IF EXISTS karyawan CASCADE;')
cur.execute('DROP TABLE IF EXISTS admin;')
cur.execute('DROP TABLE IF EXISTS absensi;')

cur.execute('CREATE TABLE karyawan (npm varchar(24) PRIMARY KEY,'
            'status varchar(16) NOT NULL,'
            'image BYTEA,'
            'email varchar(150) NOT NULL,'
            'nama varchar (150) NOT NULL,'
            'token varchar (150) NOT NULL);'
            )
cur.execute('CREATE TABLE admin (id serial PRIMARY KEY,'
            'nama varchar (150) NOT NULL,'
            'password varchar (150) NOT NULL,'
            'email varchar (150) NOT NULL);'
            )
cur.execute('CREATE TABLE absensi (id serial PRIMARY KEY,'
            'karyawan_id varchar(24) NOT NULL,'
            'waktu_masuk TIMESTAMP WITHOUT TIME ZONE,'
            'waktu_keluar TIMESTAMP WITHOUT TIME ZONE,'
            'status char(16) NOT NULL,'
            'CONSTRAINT fk_karyawan FOREIGN KEY(karyawan_id) REFERENCES karyawan(npm));'
            )

data = [
  ('1200005', 'Patrick Pratama Hendri', 'Not Attended', '1200005.patrick@email.com', 'zedkxdrnfplsuchj'),
  ('5220015', 'Sandy Putra Efendi', 'Not Attended', '5220015.sandy@email.com', 'fnbxucrruuqsozmu'),
  ('5220021', 'Sandy Alferro Dion', 'Not Attended', '5220021.sandy@email.com', 'ygebkxnuemntwhkf'),
  ('07180066', 'Herman', 'Not Attended', '7180066.herman@email.com', 'hkcywyqiiujdfjnm'),
  ('2231002', 'Jetset', 'Not Attended', '2231002.jetset@email.com', 'xdthygofxlxaqnew'),
  ('2231061', 'Erwin', 'Not Attended', '2231061.erwin@email.com', 'ecdnjcjaicqdwqaj'),
  ('2231064', 'Fernando Jose', 'Not Attended', '2231064.fernando@email.com', 'hqbqfjfzjjzzjepz'),
  ('2231065', 'Deric Cahyadi', 'Not Attended', '2231065.deric@email.com', 'ktirmfdzoybvmnzd'),
  ('2231068', 'Muhammad Arif Guntara', 'Not Attended', '2231068.muhammad@email.com', 'ahnuofyytgeutqjs'),
  ('2231098', 'Dedy Susanto', 'Not Attended', '2231098.dedy@email.com', 'uskanhqmkedzclat'),
  ('2231109', 'Wirianto', 'Not Attended', '2231109.wirianto@email.com', 'wpjtkmqyjsgdkiyv'),
  ('2231127', 'Mellberg Limanda', 'Not Attended', '2231127.mellberg@email.com', 'lubjqacedhagdwtb'),
  ('2231129', 'Christoper', 'Not Attended', '2231129.christoper@email.com', 'nyxpcneduxunvihh'),
  ('2231004', 'Brian Tracia Bahagia', 'Not Attended', '2231004.brian@email.com', 'aswflebuuyssawod'),
  ('2231007', 'Vincent Claudius Santoso', 'Not Attended', '2231007.vincent@email.com', 'rgqprrqsjyhpslzn'),
  ('2231009', 'Inov Susanto', 'Not Attended', '2231009.inov@email.com', 'npfedgljmjatptpl'),
  ('2231048', 'Paerin', 'Not Attended', '2231048.paerin@email.com', 'souuabtteyanpqqn'),
  ('2231055', 'Risna Yunita', 'Not Attended', '2231055.risna@email.com', 'drexluacuoimvkos'),
  ('2231059', 'Jannes Velando', 'Not Attended', '2231059.jannes@email.com', 'udjdyxjiqxnspubt'),
  ('2231006', 'Yulsen', 'Not Attended', '2231006.yulsen@email.com', 'hlcgxmhcxehsxxro'),
  ('2231073', 'Wilson', 'Not Attended', '2231073.wilson@email.com', 'vybqntzjbdcyzqph'),
  ('2231086', 'Chelsea', 'Not Attended', '2231086.chelsea@email.com', 'osihzmhtpsubduny'),
  ('2231089', 'Fransisco', 'Not Attended', '2231089.fransisco@email.com', 'giywaxjydmlrozca'),
  ('2231105', 'Isaac Julio Herodion', 'Not Attended', '2231105.isaac@email.com', 'fwlvhcfxbqvrqydv'),
  ('2231119', 'Rubin', 'Not Attended', '2231119.rubin@email.com', 'llusiypxidxsdsma'),
  ('2231120', 'Kevin Gernaldi', 'Not Attended', '2231120.kevin@email.com', 'upwshrboztnuuwft'),
  ('2231135', 'Darren Huang', 'Not Attended', '2231135.darren@email.com', 'qeifqrytwzonkolc'),
  ('2231163', 'Stanley', 'Not Attended', '2231163.stanley@email.com', 'nqqmpffxgzpmptzj'),
  ('2231198', 'Fajar Anugrah', 'Not Attended', '2231198.fajar@email.com', 'bbgzvaxnsinyefvt'),
  ('2231206', 'Kelvin Tang', 'Not Attended', '2231206.kelvin@email.com', 'kmuzoqckcmsotuih'),
  ('03170028', 'Wike Orize', 'Not Attended', '3170028.wike@email.com', 'sqmvpdexbaizokyv'),
  ('09200101', 'Abdurrakhman Alhakim', 'Not Attended', '09200101.abdurrakhman@email.com', 'qbomdmuwxtgguzut')
]

for record in data:
    npm, name, _, __, ___ = record
    image = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((10, 50), f"NPM: {npm}\nName: {name}", fill=(0, 0, 0))
    image.save(f"{output_dir}{npm}.png")

for record in data:
  cur.execute(
      "INSERT INTO karyawan (npm, nama, status, email, token) VALUES (%s, %s, %s, %s, %s)",
      record
  )
# for record in data:

for record in data:
  with open(f"{output_dir}{record[0]}.png", "rb") as img_file:
      binary_data = img_file.read()
      cur.execute(
          "UPDATE karyawan SET image = %s WHERE npm = %s",
          (binary_data, record[0])
      )

cur.execute(
    "INSERT INTO admin (nama, password, email) VALUES (%s, %s, %s)",
    ("admin", hashed_password.decode('utf-8'), "admin@admin.com")
)
# # Insert data into the table

# cur.execute('INSERT INTO books (title, author, pages_num, review)'
#             'VALUES (%s, %s, %s, %s)',
#             ('A Tale of Two Cities',
#              'Charles Dickens',
#              489,
#              'A great classic!')
#             )


# cur.execute('INSERT INTO books (title, author, pages_num, review)'
#             'VALUES (%s, %s, %s, %s)',
#             ('Anna Karenina',
#              'Leo Tolstoy',
#              864,
#              'Another great classic!')
#             )

conn.commit()

cur.close()
conn.close()