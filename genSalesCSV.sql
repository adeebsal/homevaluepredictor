.open --new "./Sales.db"

CREATE TABLE Prices
(
Pid         Integer ,
Price          Real ,
`Year`         text ,
Primary Key (Pid)
)
;

CREATE TABLE Locations
(
Tid       Integer,
cbd_dist    Real ,
X_coord     Real ,
Y_coord  Integer ,
Primary Key (Tid)
)
;

CREATE TABLE Characteristics
(
Pid            Integer ,
Tid            Integer ,
Home_size      Integer ,
Parcel_size    Integer ,
Beds           Integer ,
Age            Integer ,
`Pool`         Integer ,
Primary Key (Tid) ,
Foreign Key (Pid) REFERENCES Prices(Pid),
Foreign Key (Tid) REFERENCES Locations(Tid)
)
;
.tables
.schema


.import --csv ./raw/characteristics.csv Characteristics
.import --csv ./raw/prices.csv Prices
.import --csv ./raw/locations.csv Locations

SELECT * from Characteristics;

.headers ON
.mode csv
.output ./sales.csv

SELECT p.Price, p.`Year`, c.Home_size, c.Parcel_size, c.Beds, c.Age, c.`Pool`, l.X_coord, l.Y_coord, l.cbd_dist
FROM Characteristics AS c
JOIN Locations AS l on c. Tid = l.Tid
JOIN Prices AS p on c. Pid = p.Pid
GROUP By p.Price
;