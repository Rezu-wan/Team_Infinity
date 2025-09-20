#include <Servo.h>
Servo s;


 #define ENA 11

  #define IN1 12
  #define IN2 10


#define trigF 8
#define echoF 9

#define trigL 4
#define echoL 5

#define trigR 6
#define echoR 7

long duration;
int distance;

int distanceF, distanceR, distanceL;

void setup() {
  pinMode(trigF, OUTPUT);
  pinMode(trigL, OUTPUT);
  pinMode(trigR, OUTPUT);

  pinMode(echoF, INPUT);
  pinMode(echoL, INPUT);
  pinMode(echoR, INPUT);



  pinMode(ENA,OUTPUT);

pinMode(IN1,OUTPUT);
pinMode(IN2,OUTPUT);

  Serial.begin(9600);

  s.attach(3); 
}

void loop() {

//  s.write(90);


 
  distanceF = dist(trigF, echoF);
  delay(5);
  distanceL = dist(trigL, echoL);
  delay(5);
  distanceR = dist(trigR, echoR);
  delay(5);

  Serial.print("F: "); Serial.print(distanceF);
  Serial.print(" L: "); Serial.print(distanceL);
  Serial.print(" R: "); Serial.println(distanceR);

  if(distanceF > 110 && distanceF < 600) {
    s.write(95);
    delay(20);
    run(255, HIGH, LOW);
    Serial.println("Forward");
  } 
  else if(distanceF < 10){
     s.write(95);
    delay(20);
    run(255, LOW,HIGH);
    delay(500);
  }
  else if(distanceF < 110 || distanceF > 600 ) {
    if(distanceL<=300 && distanceR<=300){
        if(distanceL > distanceR ) {
          s.write(75);
          delay(20);

          run(200, HIGH, LOW); 
           Serial.println("Left");
         } 
         else if(distanceR > distanceL) { 
          s.write(115);
           delay(20);

           run(200, HIGH, LOW); 
          Serial.println("Right");
          }
         else if(distanceR == distanceL) {
           s.write(95);
           delay(20);

            run(200, HIGH, LOW); 
           Serial.println("Right");
         }
    
   
        }
       }
    
    
    else{
            run(0, HIGH, LOW); 


      // s.write(75);
      //     delay(20);

      //     run(200, HIGH, LOW); 
      //      Serial.println("Left");
     

          // s.write(115);
          //  delay(20);

          //  run(200, HIGH, LOW); 
          // Serial.println("Right");

     }
  
}



int dist(int t, int e) {
  digitalWrite(t, LOW);
  delayMicroseconds(2);

  digitalWrite(t, HIGH);
  delayMicroseconds(10);
  digitalWrite(t, LOW);

  duration = pulseIn(e, HIGH, 20000); 
  if (duration == 0) {
    return 999;
  }
  distance = duration * 0.034 / 2;
  return distance;
}




void run(int A, int ONE, int TWO)
{

      analogWrite(ENA,A );

    digitalWrite(IN1, ONE);
    digitalWrite(IN2, TWO);

}

