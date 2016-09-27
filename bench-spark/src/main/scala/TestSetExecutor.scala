// this script is currently only meant for demonstration purposes
import collection.mutable.ArrayBuffer
import collection.mutable.LinkedHashMap
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

import java.io.{File, FileInputStream}

import org.rogach.scallop._
import org.yaml.snakeyaml.Yaml 

class CLI(arguments: Seq[String]) extends org.rogach.scallop.ScallopConf(arguments) {
  val experiments = opt[String](required = true, descr="YAML file that describes the experiment" )
  verify()
}


class TestSet (obj: Object) {
}


object TestSetExecutor {

  def main(args: Array[String]) {
    val parser = new CLI(args)
    val filename = parser.experiments()
    val newParamCategory = loadYAML(filename)
    println(newParamCategory)
    println()

    var aTestSet : LinkedHashMap[String, LinkedHashMap[String, Object]] = LinkedHashMap()
    aTestSet = genNextTestSet(aTestSet, newParamCategory)
    var testSetArray = ArrayBuffer(aTestSet)
    aTestSet = genNextTestSet(aTestSet, newParamCategory)
    while (aTestSet != testSetArray(0)) {
      testSetArray.append(aTestSet)
      aTestSet = genNextTestSet(aTestSet, newParamCategory)
    }

    testSetArray.foreach{ case e => 
      println(e)
    }

  }

  def genNextTestSet(aTestSet: LinkedHashMap[String, LinkedHashMap[String, Object]], testSetsDef: LinkedHashMap[String, LinkedHashMap[String, ArrayBuffer[Object]]]) : LinkedHashMap[String, LinkedHashMap[String, Object]] = {
    var result : LinkedHashMap[String, LinkedHashMap[String, Object]] = LinkedHashMap()
    var found = false

    if (aTestSet.size <= 0) {
      testSetsDef.foreach{ case (paramCatName, paramCatVal) =>
        val newParam : LinkedHashMap[String, Object] = LinkedHashMap()
        paramCatVal.foreach{ case (paramName, paramArray) =>
          newParam.put(paramName,paramArray(0))
        }
        result.put(paramCatName, newParam)
      }
    }

    else {
      var carry = 1
      aTestSet.foreach{ case (paramCatName, paramCatVal) =>
        val newParam : LinkedHashMap[String, Object] = LinkedHashMap()
        paramCatVal.foreach{ case (paramName, paramValue) =>
          val paramArray = testSetsDef(paramCatName)(paramName)
          val count = paramArray.size
          val index = paramArray.indexOf(paramValue)
          newParam.put(paramName, paramArray((index+carry)%count))
          if (index + carry == count) {
            carry = 1
          } else {
            carry = 0
          }
        }
        result.put(paramCatName, newParam)
      }
    }
    return result
  }

  def loadYAML(filename : String) : LinkedHashMap[String, LinkedHashMap[String, ArrayBuffer[Object]]] = {
    val file = new FileInputStream(new File(filename))
    var yamlParser = new Yaml()
    var testSetDef : LinkedHashMap[String, LinkedHashMap[String, Array[Object]]] = LinkedHashMap()
    val testSetDefJava = yamlParser.load(file).asInstanceOf[java.util.LinkedHashMap[String, Object]]

    // Convert Java structure to Scala structure
    var newParamCategory : LinkedHashMap[String, LinkedHashMap[String,  ArrayBuffer[Object]]] = LinkedHashMap()
    testSetDefJava.foreach{ case(paramCategoryName, paramCategoryValues) =>
      var newParamMap : LinkedHashMap[String,  ArrayBuffer[Object]] = LinkedHashMap()
      paramCategoryValues.asInstanceOf[java.util.LinkedHashMap[String, Object]].foreach{ case(paramName, paramArray) =>
        val normalizedParamArray = paramArray match {
          case paramArray : java.lang.Integer => Seq(paramArray).asJava
          case paramArray : java.util.List[Object] => paramArray
          case _ => paramArray.getClass
        }
        var newParamArray : ArrayBuffer[Object] = ArrayBuffer()
        normalizedParamArray.asInstanceOf[java.util.List[Object]].foreach{ case(param) =>
          val newParam = param match {
            case param : String => param
            case param : java.lang.Integer => param.toString
            case param : java.util.LinkedHashMap[String, String] => mapAsScalaMapConverter(param).asScala
            case _ => param.getClass
          }
          newParamArray.append(newParam)
        }
        newParamMap.put(paramName, newParamArray)
      }
      newParamCategory.put(paramCategoryName, newParamMap)
    }
    return newParamCategory
  }

}

