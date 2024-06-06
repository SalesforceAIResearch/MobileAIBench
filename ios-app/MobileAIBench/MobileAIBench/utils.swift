//
//  utils.swift
//  LLMBench
//
//  Created by Tulika Awalgaonkar on 4/10/24.
//

import Foundation

func normalizeAnswer(_ s: String) -> String {
    func removeArticles(_ text: String) -> String {
        let regex = try! NSRegularExpression(pattern: "\\b(a|an|the)\\b", options: [])
        return regex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "")
    }
    
    func whiteSpaceFix(_ text: String) -> String {
        return text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.joined(separator: " ")
    }
    
    func removePunc(_ text: String) -> String {
        let punctuation = CharacterSet.punctuationCharacters
        return String(text.unicodeScalars.filter { !punctuation.contains($0) })
    }
    
    func lower(_ text: String) -> String {
        return text.lowercased()
    }
    let output = whiteSpaceFix(removeArticles(removePunc(lower(s))))
    //print(output)
    return output
}

func EM_metric(actual: String, predicted: String)-> Double{
    let output = normalizeAnswer(predicted) == normalizeAnswer(actual) ? 1.0 : 0.0
    //print(output)
    return output
}
func F1_metric(actual: String, predicted: String)-> Double{
    let predictionTokens = normalizeAnswer(predicted).components(separatedBy: .whitespaces)
    let groundTruthTokens = normalizeAnswer(actual).components(separatedBy: .whitespaces)
    let common = Set(predictionTokens).intersection(Set(groundTruthTokens))
    let numCommon = Double(common.count)
    if numCommon == 0 {
        return 0.0
    }
    let precision = numCommon / Double(predictionTokens.count)
    let recall = numCommon / Double(groundTruthTokens.count)
    let f1 = (2 * precision * recall) / (precision + recall)
    return f1
}
func parseSQLToDict(sql: String) -> [(String, Any)] {
    // Define patterns to match different parts of the SQL query
    let patterns: [String: NSRegularExpression] = [
        "select": try! NSRegularExpression(pattern: "SELECT\\s+(.*?)\\s+FROM", options: .caseInsensitive),
        "from": try! NSRegularExpression(pattern: "FROM\\s+(.*?)\\s+(WHERE|GROUP BY|ORDER BY|LIMIT|$)", options: .caseInsensitive),
        "where": try! NSRegularExpression(pattern: "WHERE\\s+(.*?)(GROUP BY|ORDER BY|LIMIT|$)", options: .caseInsensitive),
        "group_by": try! NSRegularExpression(pattern: "GROUP BY\\s+(.*?)(ORDER BY|LIMIT|$)", options: .caseInsensitive),
        "order_by": try! NSRegularExpression(pattern: "ORDER BY\\s+(.*?)(LIMIT|$)", options: .caseInsensitive),
        "limit": try! NSRegularExpression(pattern: "LIMIT\\s+(\\d+)", options: .caseInsensitive)
    ]

    // Initialize the result array
    var parsedDict: [String: [Any]] = ["select": [], "from": [], "where": [], "group_by": [], "order_by": [], "limit": []]

    // Extract and clean values from the SQL query based on a regex pattern
    for (key, pattern) in patterns {
        if let match = pattern.firstMatch(in: sql, options: [], range: NSRange(location: 0, length: sql.utf16.count)) {
            let range = Range(match.range(at: 1), in: sql)!
            let values = sql[range].trimmingCharacters(in: .whitespacesAndNewlines)
            // For 'select', 'from', 'group_by', and 'order_by' clauses, split by commas
            if ["select", "from", "group_by", "order_by"].contains(key) {
                parsedDict[key] = values.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            }
            // For 'where' clause, handle AND/OR
            else if key == "where" {
                let whereParts = values.components(separatedBy: #"\s+(AND|OR)\s+"#)
                parsedDict[key] = [whereParts[0].trimmingCharacters(in: .whitespacesAndNewlines)]
                for i in stride(from: 1, to: whereParts.count, by: 2) {
                    parsedDict[key]?.append(whereParts[i].uppercased())  // AND/OR
                    parsedDict[key]?.append(whereParts[i + 1].trimmingCharacters(in: .whitespacesAndNewlines))
                }
            }
            else if key == "limit" {
                parsedDict[key] = [Int(values)!]  // Convert limit to int and put in a list
            }
        }
    }

    // Convert the dictionary to an array of tuples
    var resultList: [(String, Any)] = []
    for (key, values) in parsedDict {
        for value in values {
            resultList.append((key, value))
        }
    }

    return resultList
}


func calculateF1Score(l1: [String], l2: [String]) -> Double {
    // Calculate the true positives, false positives, and false negatives
    let tp = Set(l1).intersection(Set(l2)).count
    let fp = Set(l1).subtracting(Set(l2)).count
    let fn = Set(l2).subtracting(Set(l1)).count

    let precision = Float(tp) / Float(tp + fp)
    let recall = Float(tp) / Float(tp + fn)
    let f1Score = 2 * (precision * recall) / (precision + recall)

    return Double(f1Score.isNaN ? 0.0 : f1Score)
}


func SQL_parser_metric(predictedOutput: String, actualOutput: String) -> Double {
    let gt = parseSQLToDict(sql: actualOutput)
    let pred = parseSQLToDict(sql: predictedOutput)
    // Convert each element of the tuples to String and lowercase
    let gtLowercased = gt.map { ($0.0, String(describing: $0.1).lowercased()) }
    let predLowercased = pred.map { ($0.0, String(describing: $0.1).lowercased()) }
    
    // Flatten the array of tuples into a single array of strings for comparison
    let gtFlattened = gtLowercased.flatMap { [$0.0, $0.1] }
    let predFlattened = predLowercased.flatMap { [$0.0, $0.1] }

    return calculateF1Score(l1: gtFlattened, l2: predFlattened)
}

func levenshtein_metric(actual: String, predicted: String) -> Double {
    if predicted.isEmpty {
        return 1.0
    }

    let len1 = actual.count
    let len2 = predicted.count

    // Create a matrix to store the distances
    var matrix = [[Int]](repeating: [Int](repeating: 0, count: len2 + 1), count: len1 + 1)

    // Initialize the matrix with values from 0 to len1 for the first column
    for i in 0...len1 {
        matrix[i][0] = i
    }

    // Initialize the matrix with values from 0 to len2 for the first row
    for j in 0...len2 {
        matrix[0][j] = j
    }

    // Fill in the rest of the matrix
    for i in 1...len1 {
        for j in 1...len2 {
            let actualIndex = actual.index(actual.startIndex, offsetBy: i - 1)
            let predictedIndex = predicted.index(predicted.startIndex, offsetBy: j - 1)
            
            let cost = actual[actualIndex] == predicted[predictedIndex] ? 0 : 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,                  // deletion
                matrix[i][j - 1] + 1,                  // insertion
                matrix[i - 1][j - 1] + cost           // substitution
            )
        }
    }

    let maxLen = max(len1, len2)
    return Double(matrix[len1][len2]) / Double(maxLen)
}

func rouge1_metric(actual: String, predicted: String)-> Double{
        let actualWords = actual.components(separatedBy: " ")
        let predictedWords = predicted.components(separatedBy: " ")
        let intersection = Set(actualWords).intersection(Set(predictedWords))
        let precision = Double(intersection.count) / Double(predictedWords.count)
        let recall = Double(intersection.count) / Double(actualWords.count)
        let rouge1Score = 2 * (precision * recall) / (precision + recall)
        return rouge1Score
}
func longestCommonSubsequenceLength(actual: [Character], predicted: [Character]) -> Int {
    let m = actual.count
    let n = predicted.count
    if m<1 || n<1{
        return 0
    }
    var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
    for i in 1...m {
        for j in 1...n {
            if actual[i - 1] == predicted[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }
    return dp[m][n]
}
func rougeL_metric(actual: String, predicted: String)-> Double{
        let actualChars = Array(actual)
        let predictedChars = Array(predicted)
        let lcsLength = longestCommonSubsequenceLength(actual: actualChars, predicted: predictedChars)
        let precision = Double(lcsLength) / Double(predictedChars.count)
        let recall = Double(lcsLength) / Double(actualChars.count)
        let rougeLScore = 2 * (precision * recall) / (precision + recall)
        return rougeLScore
}

func task_specific_metric(dataset: String, actual: String, predicted: String)->(Double, Double){
    var metric1=0.0
    var metric2=0.0
    if dataset=="hotpot_qa" || dataset=="databricks_dolly"{
        metric1=EM_metric(actual: actual, predicted: predicted)
        metric2=F1_metric(actual: actual, predicted: predicted)
    }
    else if dataset=="sql_create_context"{
        metric1=SQL_parser_metric(predictedOutput: predicted, actualOutput: actual)
        metric2=levenshtein_metric(actual: actual, predicted: predicted)
    }
    else if dataset=="edinburgh_xsum"{
        metric1=rouge1_metric(actual: actual, predicted: predicted)
        metric2=rougeL_metric(actual: actual, predicted: predicted)
    }
    if metric1.isNaN{
        metric1=0.0
    }
    if metric2.isNaN{
        metric2=0.0
    }
    return (metric1, metric2)
}

func print_task_specific_metric(dataset: String, metric1: Double, metric2:Double)->String{
    var fstring="\n\nTASK SPECIFIC METRIC"
    if dataset=="hotpot_qa" || dataset=="databricks_dolly"{
        fstring += """
            \nExact match: \(metric1)
            F1 score: \(metric2)
            """
    }
    else if dataset=="sql_create_context"{
        fstring += """
            \nSQL parser metric: \(metric1)
            Levenshtein metric: \(metric2)
            """
    }
    else if dataset=="edinburgh_xsum"{
        fstring += """
            \nRouge1 metric: \(metric1)
            RougeL metric: \(metric2)
            """
    }
    fstring+="\n--------------------------------------------------------\n"
    return fstring
}

func get_SYS_prompt(dataset: String, include_context: Bool)->String{
    var SYS = ""
    
    if dataset=="databricks_dolly" || dataset=="hotpot_qa"{
        if include_context==true{
            SYS = "You're a helpful assistant proficient in answering questions. The following context was used to answer the question: {context}"
        }
        else{
            SYS = "You're a helpful assistant proficient in answering questions."
        }
        
    }
    else if dataset=="sql_create_context"{
        SYS="You're a helpful assistant proficient in crafting SQL queries. The following command was used to create the table: {context}\n"
    }
    else if dataset=="edinburgh_xsum"{
        SYS = "You're a helpful assistant in summarizing articles.\n"
    }
    else{
        SYS=""
    }
    return SYS
}

func get_prompt(dataset: String, question:String, SYS:String, include_context:Bool, con:String)->(String, String){
    var new_sys=""
    var new_ques=""
    if dataset=="sql_create_context"{
        let context=con as? String ?? ""
        new_sys = SYS.replacingOccurrences(of: "{context}", with: context)
        new_ques=question
    }
    else if dataset=="edinburgh_xsum"{
        var tmp=""
        tmp = "Summarize the following article: "+question
        let limit = 1024 - (SYS.count+128)
        if tmp.count >= limit {
            tmp = String(tmp.prefix(limit))
        }
        new_ques = tmp.replacingOccurrences(of: "'", with: "\\'")+"\n"
        new_sys = SYS
    }
    else{
        if include_context==true{
            var context=con as? String ?? ""
            let limit = 1024 - (question.count+SYS.count+128)
            if context.count >= limit {
                context = String(context.prefix(limit))
            }
            new_sys = SYS.replacingOccurrences(of: "{context}", with: context)
        }
        else
        {
            new_sys=SYS
        }
        new_ques=question
    }
    return (new_sys, new_ques)
}
