name := "sgdmf"
version := "1.0"
scalaVersion := "2.11.0"
//scalaVersion := "2.10.2"


// http://mvnrepository.com/artifact/org.apache.commons/commons-math3
libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "1.6.0"
libraryDependencies += "org.apache.commons" % "commons-math3" % "3.0"

// http://mvnrepository.com/artifact/org.scalanlp/breeze_2.10
libraryDependencies += "org.scalanlp" % "breeze_2.11" % "0.12"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.0"
